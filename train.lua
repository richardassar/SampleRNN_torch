require 'nn'
require 'rnn'
require 'nngraph'
require 'gnuplot'
require 'image'
require 'optim'
require 'cunn'
require 'cudnn'
require 'audio'

local max_iter = 100
local max_grad_norm = 0.1

local big_frame_size = 8
local frame_size = 2

local big_dim = 1024
local dim = big_dim
local q_levels = 256
local emb_size = 256

local dat = audio.load('piano_note.wav'):view(-1)[{{1,1024}}]--torch.sin(i*4) + torch.sin(i*32)*0.2
dat:csub(dat:min())
dat:div(dat:max())

local q_dat = 1 + torch.floor(dat * (q_levels - 1))

local big_input_sequences = -2 + 4 * (q_dat[{{1,-1-big_frame_size}}] - 1) / (q_levels - 1)
local input_sequences = -2 + 4 * (q_dat[{{big_frame_size-frame_size+1,-1-frame_size}}] - 1) / (q_levels - 1)
local target_sequences = q_dat[{{big_frame_size+1,-1}}]
local prev_samples = q_dat[{{big_frame_size-frame_size+1,-1-1}}]

local big_frames = big_input_sequences:unfold(1,big_frame_size,big_frame_size)
local frames = input_sequences:unfold(1,frame_size,frame_size)
prev_samples = prev_samples:unfold(1,frame_size,1)

function big_frame_level_rnn(inp)
    local L1 = inp 
             - nn.View(1,-1):setNumInputDims(1)
             - cudnn.GRU(big_frame_size, big_dim, 1)
             - nn.View(-1):setNumInputDims(1)

    local output = L1
                 - nn.Linear(big_dim, dim * big_frame_size / frame_size)
                 - nn.View(-1,dim)

    local independent_preds = L1
                            - nn.Linear(big_dim, q_levels * big_frame_size)
                            - nn.View(-1,q_levels)

    return output, independent_preds
end

function frame_level_rnn(inp, other_inp)
    local L2 = inp 
             - nn.Linear(frame_size, dim)    
            
    local output = nn.CAddTable()({L2, other_inp})
                 - nn.View(1,-1):setNumInputDims(1)
                 - cudnn.GRU(dim, dim, 1)
                 - nn.View(-1):setNumInputDims(1)
                 - nn.Linear(dim, dim * frame_size)
                 - nn.View(-1,dim)

    return output
end

function sample_level_predictor(frame_level_outputs, prev_samples)
    local L3 = prev_samples
             - nn.Contiguous() 
             - nn.View(-1,1)
             - nn.LookupTable(q_levels, emb_size)           
             - nn.View(-1,frame_size*emb_size)
             - nn.Linear(frame_size*emb_size, dim, false)

    local output = nn.CAddTable()({L3, frame_level_outputs})
                 - nn.Linear(dim,dim)
                 - cudnn.Tanh()
                 - nn.Linear(dim,dim)
                 - cudnn.Tanh()
                 - nn.Linear(dim,q_levels)
                 - cudnn.SoftMax()

    return output
end

local i1 = nn.Identity()()
local i2 = nn.Identity()()
local i3 = nn.Identity()()

local big_frame_outputs,independent_preds = big_frame_level_rnn(i1)
local frame_level_outputs = frame_level_rnn(i2, big_frame_outputs)
local sample_level_outputs = sample_level_predictor(frame_level_outputs, i3)

local net = nn.gModule({i1,i2,i3},{sample_level_outputs,independent_preds}):cuda()
local param,dparam = net:getParameters()

local linearLayers = net:findModules('nn.Linear')
for k,v in pairs(linearLayers) do
    v:reset(math.sqrt(2/(v.weight:size(2))))
end

function resetStates()
    local lstms = net:findModules('cudnn.GRU')
    for i=1,#lstms do
        lstms[i]:resetStates()
    end
end

local criterion = nn.ParallelCriterion()
                    :add(nn.CrossEntropyCriterion())
                    :add(nn.CrossEntropyCriterion())
                    :cuda()

local inp = {big_frames:cuda(),frames:cuda(),prev_samples:cuda()}
local out = {target_sequences:cuda(),target_sequences:cuda()}

local optim_config = {
    learningRate = 0.001,
}

local loss = torch.DoubleTensor(max_iter)

for i = 1,max_iter do
    function feval(x)
        if x ~= param then
            param:copy(x)
        end

        net:zeroGradParameters()

        resetStates()

        local f = criterion:forward(net:forward(inp),out)
        net:backward(inp,criterion:backward(net.output,out))

        local grad_norm = dparam:norm(2)
        if grad_norm > max_grad_norm then
            print(grad_norm)
            local shrink_factor = max_grad_norm / grad_norm
            dparam:mul(shrink_factor)
        end

        return f,dparam
    end

    local _, err = optim.adam(feval,param,optim_config)
    loss[i] = err[1]

    image.save(string.format('vis/%05d.png', i), image.vflip(image.toDisplayTensor(net.output[1]:t())))

    print(i..': '..err[1])
end

val,idx = torch.max(net.output[1],2)

gnuplot.pngfigure('plot_labels.png')
gnuplot.plot({'True',target_sequences,'-'}, {'Predicted',idx,'-'})
gnuplot.plotflush()

image.save('probs.png', image.vflip(image.toDisplayTensor(net.output[1]:t())))

gnuplot.pngfigure('loss_curve.png')
gnuplot.plot(loss,'-')
gnuplot.plotflush()