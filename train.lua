require 'nn'
require 'rnn'
require 'gnuplot'
require 'image'
require 'optim'
require 'cunn'
require 'cudnn'
require 'audio'
cudnn.fastest = true

--
local learning_rate = 0.001
local max_iter = 1000
local max_grad = 1
--local max_grad_norm = 0.1

local big_frame_size = 64
local frame_size = 16

local big_dim = 1024
local dim = big_dim
local q_levels = 256
local emb_size = 256

local sample_length = 8000*5

--
local dat = audio.load('piano_note.wav'):view(-1)[{{1,2048}}]
dat:csub(dat:min())
dat:div(dat:max())

local q_dat = 1 + torch.floor(dat * (q_levels - 1))

local big_input_sequences = q_dat[{{1,-1-big_frame_size}}]
local input_sequences = q_dat[{{big_frame_size-frame_size+1,-1-frame_size}}]
local target_sequences = q_dat[{{big_frame_size+1,-1}}]
local prev_samples = q_dat[{{big_frame_size-frame_size+1,-1-1}}]

local big_frames = big_input_sequences:unfold(1,big_frame_size,big_frame_size)
local frames = input_sequences:unfold(1,frame_size,frame_size)
prev_samples = prev_samples:unfold(1,frame_size,1)

local big_frame_level_rnn = nn.Sequential()
    :add(nn.AddConstant(-1))    
    :add(nn.MulConstant(4/(q_levels-1)))
    :add(nn.AddConstant(-2))
    :add(nn.View(1,-1):setNumInputDims(1))
    :add(cudnn.GRU(big_frame_size, big_dim, 1))
    :add(nn.View(-1):setNumInputDims(1))
    :add(nn.ConcatTable()
        :add(nn.Sequential()
            :add(nn.Linear(big_dim, dim * big_frame_size / frame_size))
            :add(nn.View(-1,dim))
        )
        :add(nn.Sequential()
            :add(nn.Linear(big_dim, q_levels * big_frame_size))
            :add(nn.View(-1,q_levels))
        )
    )

local frame_level_rnn = nn.Sequential()
    :add(nn.ParallelTable()
        :add(nn.Sequential()
            :add(nn.AddConstant(-1))
            :add(nn.MulConstant(4/(q_levels-1)))
            :add(nn.AddConstant(-2))
            :add(nn.Linear(frame_size, dim))
        )
        :add(nn.Identity())
    )
    :add(nn.CAddTable())
    :add(nn.View(1,-1):setNumInputDims(1))
    :add(cudnn.GRU(dim, dim, 1))
    :add(nn.View(-1):setNumInputDims(1))
    :add(nn.Linear(dim, dim * frame_size))
    :add(nn.View(-1,dim))


local sample_level_predictor = nn.Sequential()
    :add(nn.ParallelTable()
        :add(nn.Identity())
        :add(nn.Sequential()
            :add(nn.Contiguous())
            :add(nn.View(-1,1))
            :add(nn.LookupTable(q_levels, emb_size))
            :add(nn.View(-1,frame_size*emb_size))
            :add(nn.Linear(frame_size*emb_size, dim, false))
        )
    )
    :add(nn.CAddTable())
    :add(nn.Linear(dim,dim))
    :add(cudnn.Tanh())
    :add(nn.Linear(dim,dim))
    :add(cudnn.Tanh())
    :add(nn.Linear(dim,q_levels))
    :add(cudnn.SoftMax())

local net = nn.Sequential()
    :add(nn.ParallelTable()
        :add(big_frame_level_rnn)
        :add(nn.Identity())
        :add(nn.Identity())
    )
    :add(nn.ConcatTable()
        :add(nn.Sequential()
            :add(nn.ConcatTable()                
                :add(nn.SelectTable(2))
                :add(nn.Sequential()
                    :add(nn.SelectTable(1))
                    :add(nn.SelectTable(1))
                )
            )
            :add(frame_level_rnn)
        )
        :add(nn.Identity())
    )
    :add(nn.ConcatTable()
        :add(nn.Sequential()
            :add(nn.ConcatTable()
                :add(nn.SelectTable(1))
                :add(nn.Sequential()
                    :add(nn.SelectTable(2))
                    :add(nn.SelectTable(3))
                )
            )
            :add(sample_level_predictor)
        )
        :add(nn.Sequential()
            :add(nn.SelectTable(2))
            :add(nn.SelectTable(1))
            :add(nn.SelectTable(2))
        )
    )
    :cuda()

local param,dparam = net:getParameters()

local linearLayers = net:findModules('nn.Linear')
for k,v in pairs(linearLayers) do
    v:reset(math.sqrt(2/(v.weight:size(2))))
end

function resetStates()
    local grus = net:findModules('cudnn.GRU')
    for i=1,#grus do
        grus[i]:resetStates()
    end
end

function train()
    local criterion = nn.ParallelCriterion()
        :add(nn.CrossEntropyCriterion())
        :add(nn.CrossEntropyCriterion())
        :cuda()

    local inp = {big_frames:cuda(),frames:cuda(),prev_samples:cuda()}
    local out = {target_sequences:cuda(),target_sequences:cuda()}

    local optim_config = {
        learningRate = learning_rate,
    }

    local losses = torch.DoubleTensor(max_iter)
    for i = 1,max_iter do
        function feval(x)
            if x ~= param then
                param:copy(x)
            end

            net:zeroGradParameters()

            resetStates()

            local pred = net:forward(inp)
            local loss = criterion:forward(pred,out)
            net:backward(inp,criterion:backward(pred,out))

            --[[local grad_norm = dparam:norm(2)
            if grad_norm > max_grad_norm then
                print(grad_norm)
                local shrink_factor = max_grad_norm / grad_norm
                dparam:mul(shrink_factor)
            end]]--

            dparam:clamp(-max_grad, max_grad)

            return loss,dparam
        end

        local _, err = optim.adam(feval,param,optim_config)
        losses[i] = err[1]

        image.save(string.format('vis/%05d.png', i), image.vflip(image.toDisplayTensor(net.output[1]:t())))

        print(i..': '..err[1])
    end

    val,idx = torch.max(net.output[1],2)

    gnuplot.pngfigure('plot_labels.png')
    gnuplot.raw("set terminal pngcairo size 1280, 768")
    gnuplot.plot({'True',target_sequences,'-'}, {'Predicted',idx,'-'})
    gnuplot.plotflush()

    image.save('probs.png', image.vflip(image.toDisplayTensor(net.output[1]:t())))

    gnuplot.pngfigure('loss_curve.png')
    gnuplot.plot(losses,'-')
    gnuplot.plotflush()
end

function sample()
    print("Sampling...")

    resetStates()

    local samples = torch.CudaTensor(sample_length):fill(0)
    local big_frame_level_outputs, frame_level_outputs

    samples[{{1,big_frame_size}}] = math.floor(q_levels / 2)

    local start_time = sys.clock()
    for t = big_frame_size + 1, sample_length do
        if (t-1) % big_frame_size == 0 then
            local big_frames = samples[{{t - big_frame_size, t - 1}}]:view(1,-1)        
            big_frame_level_outputs = big_frame_level_rnn:forward(big_frames)[1]            
        end        

        if (t-1) % frame_size == 0 then
            local frames = samples[{{t - frame_size, t - 1}}]:view(1,-1)        
            local _t = (((t-1) / frame_size) % (big_frame_size / frame_size)) + 1

            frame_level_outputs = frame_level_rnn:forward({frames, big_frame_level_outputs[{{_t}}]})
        end

        local prev_samples = samples[{{t - frame_size, t - 1}}]

        local _t = (t-1) % frame_size + 1        
        local sample = sample_level_predictor:forward({frame_level_outputs[{{_t}}], prev_samples})
        sample = torch.multinomial(sample,1)
        
        --print(string.format('%d: %d', t, sample[{1,1}]))
        samples[t] = sample
    end
    local stop_time = sys.clock()

    print("Generated "..(sample_length / 8000).." seconds of audio in "..(stop_time - start_time).." seconds.")

    gnuplot.pngfigure('sample_output.png')
    gnuplot.raw("set terminal pngcairo size 1280, 768")
    gnuplot.plot(samples,'-')
    gnuplot.plotflush()

    local audioOut = -0x80000000 + 0xFFFF0000 * (samples - 1) / (q_levels - 1)
    audio.save("sample.wav", audioOut:view(-1,1):double(), 8000)
end

train()
sample()