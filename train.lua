require 'xlua'
require 'nn'
require 'gnuplot'
require 'image'
require 'optim'
require 'cunn'
require 'cudnn'
require 'audio'

--
local dat,sample_rate = audio.load('backup/BeethovenSonata1.wav')

local max_iter = 1000
local batch_size = 128
local learning_rate = 0.001
local max_grad = 1
--local max_grad_norm = 0.1

local seg_len = 8*sample_rate
local seq_len = 512
local big_frame_size = 8
local frame_size = 2

local big_dim = 1024
local dim = big_dim
local q_levels = 256
local emb_size = 256

local n_samples = 5
local sample_length = 25*sample_rate

--
dat = dat:mean(2):squeeze() -- stereo to mono
dat:csub(dat:min())
dat:div(dat:max())

dat = dat:unfold(1,seg_len,1*sample_rate):contiguous()
--dat:csub(dat:min(2):expandAs(dat))
--dat:cdiv(dat:max(2):expandAs(dat))

dat = dat:unfold(2,seq_len+big_frame_size,seq_len)
dat = dat:transpose(1,2)

local q_dat = 1 + torch.floor(dat * (q_levels - 1))

local big_input_sequences = q_dat[{{},{},{1,-1-big_frame_size}}]
local input_sequences = q_dat[{{},{},{big_frame_size-frame_size+1,-1-frame_size}}]
local target_sequences = q_dat[{{},{},{big_frame_size+1,-1}}]
local prev_samples = q_dat[{{},{},{big_frame_size-frame_size+1,-1-1}}]

local big_frames = big_input_sequences:unfold(3,big_frame_size,big_frame_size)
local frames = input_sequences:unfold(3,frame_size,frame_size)
prev_samples = prev_samples:unfold(3,frame_size,1)

local big_rnn = cudnn.GRU(big_frame_size, big_dim, 1, true,false,true)
local frame_rnn = cudnn.GRU(dim, dim, 1, true,false,true)

local big_frame_level_rnn = nn.Sequential()
    :add(nn.AddConstant(-1))    
    :add(nn.MulConstant(4/(q_levels-1)))
    :add(nn.AddConstant(-2))    
    :add(big_rnn)
    :add(nn.Contiguous())
    :add(nn.ConcatTable()
        :add(nn.Sequential()
            :add(nn.Bottle(nn.Linear(big_dim, dim * big_frame_size / frame_size)))
            :add(nn.View(-1,dim):setNumInputDims(2))
        )
        :add(nn.Sequential()
            :add(nn.Bottle(nn.Linear(big_dim, q_levels * big_frame_size)))
            :add(nn.View(-1,q_levels):setNumInputDims(2))
            :add(nn.Bottle(cudnn.LogSoftMax()))
        )
    )

local frame_level_rnn = nn.Sequential()
    :add(nn.ParallelTable()
        :add(nn.Sequential()
            :add(nn.AddConstant(-1))
            :add(nn.MulConstant(4/(q_levels-1)))
            :add(nn.AddConstant(-2))
            :add(nn.Bottle(nn.Linear(frame_size, dim)))
        )
        :add(nn.Identity())
    )
    :add(nn.CAddTable())
    :add(frame_rnn)
    :add(nn.Contiguous())
    :add(nn.Bottle(nn.Linear(dim, dim * frame_size)))
    :add(nn.View(-1,dim):setNumInputDims(2))
    
local sample_level_predictor = nn.Sequential()
    :add(nn.ParallelTable()
        :add(nn.Identity())
        :add(nn.Sequential()
            :add(nn.Bottle(nn.LookupTable(q_levels, emb_size),2,3)) -- TODO: Use https://github.com/facebook/fbcunn/blob/master/src/LookupTableGPU.cu
            :add(nn.View(-1,frame_size*emb_size):setNumInputDims(3))
            :add(nn.Bottle(nn.Linear(frame_size*emb_size, dim, false)))
        )
    )
    :add(nn.CAddTable())
    :add(nn.Bottle(
        nn.Sequential()
            :add(nn.Linear(dim,dim))
            :add(cudnn.ReLU())
            :add(nn.Linear(dim,dim))
            :add(cudnn.ReLU())
            :add(nn.Linear(dim,q_levels))
            :add(cudnn.LogSoftMax())        
    ))    
    
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

local gpus = torch.range(1, cutorch.getDeviceCount()):totable()
net = nn.DataParallelTable(1,true,false):add(net,gpus):threads(function() -- TODO: optional nccl
    local cudnn = require 'cudnn'
end):cuda()

local linearLayers = net:findModules('nn.Linear')
for k,v in pairs(linearLayers) do
    v:reset(math.sqrt(2/(v.weight:size(2))))
end

function resetStates(model)
    if model.impl then
        model.impl:exec(function(m)
            local grus = m:findModules('cudnn.GRU')
            for i=1,#grus do
                grus[i]:resetStates()
            end
        end)
    else
        local grus = model:findModules('cudnn.GRU')
        for i=1,#grus do
            grus[i]:resetStates()            
        end
    end

    model:clearState()
end

function train(net)
    net:training()

    local param,dparam = net:getParameters()

    local crit1 = nn.ClassNLLCriterion()
    local crit2 = nn.ClassNLLCriterion()

    local criterion = nn.ParallelCriterion()
        :add(crit1)
        :add(crit2)
        :cuda()

    local optim_state = torch.load("optim_state.t7")
    --[[local optim_state = {
        learningRate = learning_rate
    }]]--

    local losses = torch.DoubleTensor(max_iter)
    for i = 1,max_iter do
        local total_err = 0
        
        local batches = {}
        local ord=torch.randperm(big_frames:size(2)):long()

        local start=1
        while start<=big_frames:size(2) do
            local stop = start+batch_size-1
            if stop>big_frames:size(2) then
                break -- Smaller batches seem to hurt the network
                stop = big_frames:size(2)
            end

            batches[#batches + 1] = ord[{{start,stop}}]
            start=stop+1
        end

        for j=1,#batches do        
            local k=batches[j]

            print("Batch "..j.."/"..#batches)
            
            resetStates(net)       
            for t=1,big_frames:size(1) do    
                local _big_frames = big_frames:select(1,t):index(1,k):cuda()
                local _frames = frames:select(1,t):index(1,k):cuda()
                local _prev_samples = prev_samples:select(1,t):index(1,k):cuda()

                local inp = {_big_frames,_frames,_prev_samples}
                local targets = target_sequences:select(1,t):index(1,k):contiguous():view(-1):cuda()

                local out = {targets,targets}
                    
                function feval(x)
                    if x ~= param then                        
                        param:copy(x)
                        net:syncParameters()
                    end

                    net:zeroGradParameters()
            
                    local pred = net:forward(inp)    
                    pred[1]=pred[1]:view(-1,q_levels)
                    pred[2]=pred[2]:view(-1,q_levels)
                    
                    local loss = criterion:forward(pred,out)
                    print(t.."/"..big_frames:size(1), loss, crit1.output * math.log(math.exp(1),2))

                    local grad = criterion:backward(net.output,out)
                    grad[1]=grad[1]:view(inp[3]:size(1),inp[3]:size(2),q_levels)
                    grad[2]=grad[2]:view(inp[3]:size(1),inp[3]:size(2),q_levels)                

                    net:backward(inp,grad)

                    --[[local grad_norm = dparam:norm(2)
                    if grad_norm > max_grad_norm then
                        --print(grad_norm)
                        local shrink_factor = max_grad_norm / grad_norm
                        dparam:mul(shrink_factor)
                    end]]--

                    dparam:clamp(-max_grad, max_grad)
                    
                    return loss,dparam                    
                end

                local _, err = optim.adam(feval,param,optim_state)
                total_err = total_err + err[1]
            end            
        end

        local err = total_err -- TODO: nats * math.log(math.exp(1),2)
        losses[i] = err

        torch.save("optim_state.t7", optim_state)        
        torch.save("net.t7",net)
            
        image.save(string.format('vis/%05d.png', i), image.vflip(image.toDisplayTensor(torch.exp(net.output[1]):t())))

        gnuplot.pngfigure('loss_curve.png')
        gnuplot.plot(losses[{{1,i}}],'-')
        gnuplot.plotflush()

        print('Iter: '..i..', loss = '..err)

        path.mkdir(string.format('samples/%d/',i))
        sample(net,string.format('samples/%d/sample.wav',i))
    end
end

function sample(net,filepath)
    print("Sampling...")

    local big_frame_level_rnn = net:get(1):get(1):get(1)
    local frame_level_rnn = net:get(1):get(2):get(1):get(2)
    local sample_level_predictor = net:get(1):get(3):get(1):get(2)
    local big_rnn = big_frame_level_rnn:get(4)
    local frame_rnn = frame_level_rnn:get(3)

    net:evaluate()
    resetStates(net)

    local samples = torch.CudaTensor(n_samples, 1, sample_length):fill(0)
    local big_frame_level_outputs, frame_level_outputs

    samples[{{},{},{1,big_frame_size}}] = math.floor(q_levels / 2) -- Silence
    --samples[{{},{},{1,big_frame_size}}] = torch.floor(torch.rand(n_samples,1,big_frame_size)*q_levels) -- Uniform noise
    --samples[{{},{},{1,big_frame_size}}] = q_dat[{{2},{1},{1,big_frame_size}}]:expandAs(samples[{{},{},{1,big_frame_size}}]) -- A snippet of a sample from the training set

    --[[big_rnn.cellInput = torch.rand(1, n_samples, big_dim):cuda() - 0.5 -- Randomise the RNN initial state state
    big_rnn.hiddenInput = torch.rand(1, n_samples, big_dim):cuda() - 0.5
    frame_rnn.cellInput = torch.rand(1, n_samples, big_dim):cuda() - 0.5
    frame_rnn.hiddenInput = torch.rand(1, n_samples, big_dim):cuda() - 0.5]]--

    local start_time = sys.clock()

    for t = big_frame_size + 1, sample_length do
        if (t-1) % big_frame_size == 0 then
            local big_frames = samples[{{},{},{t - big_frame_size, t - 1}}]
            big_frame_level_outputs = big_frame_level_rnn:forward(big_frames)[1]                    
        end        

        if (t-1) % frame_size == 0 then
            local frames = samples[{{},{},{t - frame_size, t - 1}}]
            local _t = (((t-1) / frame_size) % (big_frame_size / frame_size)) + 1

            frame_level_outputs = frame_level_rnn:forward({frames, big_frame_level_outputs[{{},{_t}}]})
        end

        local prev_samples = samples[{{},{},{t - frame_size, t - 1}}]
        
        local _t = (t-1) % frame_size + 1        

        local inp = {frame_level_outputs[{{},{_t}}], prev_samples:contiguous()}
        
        local sample = sample_level_predictor:forward(inp)        
        --sample:div(1.1) -- Sampling temperature
        sample:exp()
        sample = torch.multinomial(sample:squeeze(),1)
        
        samples[{{},1,t}] = sample

        xlua.progress(t-big_frame_size,sample_length-big_frame_size)
    end

    local stop_time = sys.clock()
    print("Generated "..(sample_length / sample_rate * n_samples).." seconds of audio in "..(stop_time - start_time).." seconds.")

    local audioOut = -0x80000000 + 0xFFFF0000 * (samples - 1) / (q_levels - 1)
    for i=1,audioOut:size(1) do
        audio.save(filepath:gsub(".wav",string.format("_%d.wav",i)), audioOut:select(1,i):t():double(), sample_rate)
    end

    print("Audio saved.")

    net:training()
end

net=torch.load("net.t7")
train(net)