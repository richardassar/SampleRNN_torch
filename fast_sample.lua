--[[
MIT License

Copyright (c) 2017 Richard Assar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
]]--

require 'rnn'
require 'cudnn'
require 'audio'
require 'LinearWeightNorm' -- https://github.com/torch/nn/pull/1162
require 'SeqGRU_WN'
require 'utils'

--
local cmd = torch.CmdLine()
cmd:text('fast_sample.lua - Samples a model generating a single audio file')
cmd:text('')

cmd:text('Session:')
cmd:option('-session','default','The name of the session in which to locate the model to be sampled')
cmd:text('')

cmd:text('Sampling:')
cmd:option('-sample_length',20,'The duration of generated samples')
cmd:option('-sampling_temperature',1,'The sampling temperature')
cmd:text('')

cmd:text('Output:')
cmd:option('-output_path','sample.wav','The path of the output audio file')
cmd:text('')

local args = cmd:parse(arg)

assert(args.session:len() > 0, "session must be provided")

local session_path = 'sessions/'..args.session
local session = torch.load(session_path..'/session.t7')

for k,v in pairs(session) do
    args[k] = v
end

--
local linear_type = args.linear_type
local cudnn_rnn = args.cudnn_rnn

local big_frame_size = args.big_frame_size
local frame_size = args.frame_size
local big_dim = args.hidden_dim
local dim = big_dim
local q_levels = args.q_levels
local q_zero = math.floor(q_levels / 2)
local q_type = args.q_type or 'linear'
local emb_size = args.embedding_size
local dropout = args.dropout

local audio_data_path = 'datasets/'..args.dataset..'/data'
local aud,sample_rate = audio.load(audio_data_path..'/p0001.wav')

local sample_length = args.sample_length*sample_rate
local sampling_temperature = args.sampling_temperature

local output_path = args.output_path

--
local big_rnn, frame_rnn
if cudnn_rnn then
    big_rnn = cudnn.GRU(big_frame_size, big_dim, 1, false, dropout, true)
    frame_rnn = cudnn.GRU(dim, dim, 1, false, dropout, true)
else 
    big_rnn = nn.SeqGRU_WN(big_frame_size, big_dim)
    frame_rnn = nn.SeqGRU_WN(dim, dim)

    big_rnn:remember('both')
    frame_rnn:remember('both')
end

local linearType = linear_type == 'WN' and 'LinearWeightNorm' or 'Linear'
local LinearLayer = nn[linearType]

local big_frame_level_rnn = nn.Sequential()
    :add(nn.AddConstant(-1))
    :add(nn.MulConstant(4/(q_levels-1)))
    :add(nn.AddConstant(-2))
    :add(nn.View(1,-1):setNumInputDims(1))
    :add(big_rnn)
    :add(nn.View(-1):setNumInputDims(1))
    :add(LinearLayer(big_dim, dim * big_frame_size / frame_size))
    :add(nn.View(-1,dim):setNumInputDims(2))
    
local frame_level_rnn = nn.Sequential()    
    :add(nn.ParallelTable()
        :add(nn.Identity())
        :add(nn.Sequential()
            :add(nn.AddConstant(-1))
            :add(nn.MulConstant(4/(q_levels-1)))
            :add(nn.AddConstant(-2))
            :add(nn.Contiguous())
            :add(LinearLayer(frame_size, dim))
        )
    )
    :add(nn.CAddTable())
    :add(nn.View(1,-1):setNumInputDims(1))
    :add(frame_rnn)
    :add(nn.View(-1):setNumInputDims(1))
    :add(LinearLayer(dim, dim * frame_size))
    :add(nn.View(-1,dim):setNumInputDims(2))
    
local sample_level_predictor = nn.Sequential()
    :add(nn.ParallelTable()
        :add(nn.Identity())
        :add(nn.Sequential()
            :add(nn.Contiguous())
            :add(nn.View(1,-1))
            :add(nn.LookupTable(q_levels, emb_size))
            :add(nn.View(-1,frame_size*emb_size))
            :add(LinearLayer(frame_size*emb_size, dim, false))
        )
    )
    :add(nn.CAddTable())
    :add(nn.Bottle(nn.Sequential()
        :add(LinearLayer(dim,dim))
        :add(cudnn.ReLU())
        :add(LinearLayer(dim,dim))
        :add(cudnn.ReLU())
        :add(LinearLayer(dim,q_levels))
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
                :add(nn.SelectTable(1))
                :add(nn.SelectTable(2))
            )
            :add(frame_level_rnn)
        )
        :add(nn.SelectTable(3))
    )
    :add(sample_level_predictor)
    :cuda()

local param,dparam = net:getParameters()
param:copy(torch.load(session_path.."/params.t7"))

cudnn.GRU.forget = cudnn.GRU.resetStates

function resetStates()
    local grus = net:findModules('cudnn.GRU')
    for i=1,#grus do
        grus[i]:forget()
    end
end

function sample()
    print("Sampling...")

    net:evaluate()
    resetStates()

    local samples = torch.CudaTensor(sample_length):fill(0)
    local big_frame_level_outputs, frame_level_outputs

    samples[{{1,big_frame_size}}] = q_zero

    local start_time = sys.clock()
    for t = big_frame_size + 1, sample_length do
        if (t-1) % big_frame_size == 0 then
            local big_frames = samples[{{t - big_frame_size, t - 1}}]:view(1,-1)        
            big_frame_level_outputs = big_frame_level_rnn:forward(big_frames)
        end        

        if (t-1) % frame_size == 0 then
            local frames = samples[{{t - frame_size, t - 1}}]:view(1,-1)        
            local _t = (((t-1) / frame_size) % (big_frame_size / frame_size)) + 1

            frame_level_outputs = frame_level_rnn:forward({big_frame_level_outputs[{{_t}}],frames})
        end

        local prev_samples = samples[{{t - frame_size, t - 1}}]

        local _t = (t-1) % frame_size + 1
        local inp = {frame_level_outputs[{{_t}}], prev_samples}

        local sample = sample_level_predictor:forward(inp)
        sample:div(sampling_temperature)
        sample:exp()
        sample = torch.multinomial(sample,1)
        
        samples[t] = sample:typeAs(samples)

        xlua.progress(t-big_frame_size,sample_length-big_frame_size)
    end
    local stop_time = sys.clock()

    print("Generated "..(sample_length / sample_rate).." seconds of audio in "..(stop_time - start_time).." seconds.")

    if q_type == 'mu-law' then
        samples = mu2linear(samples - 1)
        samples:add(1)
        samples:div(2)
    elseif q_type == 'linear' then
        samples = (samples - 1) / (q_levels - 1)
    end

    local audioOut = -0x80000000 + 0xFFFF0000 * samples
    audio.save(output_path, audioOut:view(-1,1):double(), sample_rate)
end

sample()