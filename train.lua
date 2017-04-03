require 'nn'
require 'cunn'
require 'cudnn'
require 'rnn'
require 'LinearWeightNorm' -- PR submitted for this module
require 'optim'
require 'audio'
require 'image'
require 'gnuplot'
require 'xlua'
require 'SeqGRU_WN'

local threads = require 'threads'
threads.serialization('threads.sharedserialize')

--
local data_path = './dataset/'

local n_threads = 128
local max_epoch = 1000
local minibatch_size = 128
local learning_rate = 0.001
local max_grad = 1

local sample_rate = 16000
local seg_len = 8 * sample_rate
local seq_len = 512

local big_frame_size = 8
local frame_size = 2
local big_dim = 1024
local dim = big_dim
local q_levels = 256
local q_zero = math.floor(q_levels / 2)
local emb_size = 256

local n_samples = 5
local sample_length = 20*sample_rate

--
function create_samplernn()
    --local big_rnn = cudnn.GRU(big_frame_size, big_dim, 1, true, false, true) -- TODO: Optional dropout
    --local frame_rnn = cudnn.GRU(dim, dim, 1, true, false, true)

    local big_rnn = nn.SeqGRU_WN(big_frame_size, big_dim)
    local frame_rnn = nn.SeqGRU_WN(dim, dim)

    big_rnn:remember('both')
    frame_rnn:remember('both')

    big_rnn.batchfirst = true
    frame_rnn.batchfirst = true

    local linearType = 'LinearWeightNorm'
    local LinearLayer = nn[linearType]

    local big_frame_level_rnn = nn.Sequential()
        :add(nn.AddConstant(-1))
        :add(nn.MulConstant(4/(q_levels-1)))
        :add(nn.AddConstant(-2))
        :add(big_rnn)
        :add(nn.Contiguous())
        :add(nn.Bottle(LinearLayer(big_dim, dim * big_frame_size / frame_size)))
        :add(nn.View(-1,dim):setNumInputDims(2))
        
    local frame_level_rnn = nn.Sequential()    
        :add(nn.ParallelTable()
            :add(nn.Identity())
            :add(nn.Sequential()
                :add(nn.AddConstant(-1))
                :add(nn.MulConstant(4/(q_levels-1)))
                :add(nn.AddConstant(-2))
                :add(nn.Contiguous())
                :add(nn.Bottle(LinearLayer(frame_size, dim)))
            )
        )
        :add(nn.CAddTable())
        :add(frame_rnn)
        :add(nn.Contiguous())        
        :add(nn.Bottle(LinearLayer(dim, dim * frame_size)))
        :add(nn.View(-1,dim):setNumInputDims(2))
        
    local sample_level_predictor = nn.Sequential()
        :add(nn.ParallelTable()
            :add(nn.Identity())
            :add(nn.Sequential()
                :add(nn.Contiguous())
                :add(nn.Bottle(nn.LookupTable(q_levels, emb_size),2,3))
                :add(nn.View(-1,frame_size*emb_size):setNumInputDims(3))
                :add(nn.Bottle(LinearLayer(frame_size*emb_size, dim, false)))
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

    local linearLayers = net:findModules('nn.'..linearType)
    for _,linear in pairs(linearLayers) do        
        if linear.weight:size(1) == q_levels then
            linear:reset(math.sqrt(1 / linear.weight:size(2))) -- 'LeCunn' initialization
        else
            linear:reset(math.sqrt(2 / linear.weight:size(2))) -- 'He' initialization
        end

        if linear.bias then
            linear.bias:zero()
        end
    end

    --[[local rnns = net:findModules('cudnn.GRU')
    for _,gru in pairs(rnns) do
        local biases = gru:biases()[1]
        for k,v in pairs(biases) do
            v:zero()
        end

        local weights = gru:weights()[1]

        local stdv = math.sqrt(1 / gru.inputSize) * math.sqrt(3) -- 'LeCunn' initialization
        weights[1]:uniform(-stdv, stdv) 
        weights[2]:uniform(-stdv, stdv) 
        weights[3]:uniform(-stdv, stdv)

        stdv = math.sqrt(1 / gru.hiddenSize) * math.sqrt(3)
        weights[4]:uniform(-stdv, stdv) 
        weights[5]:uniform(-stdv, stdv) 
        
        function ortho(inputDim,outputDim)
            local rand = torch.randn(outputDim,inputDim)
            local q,r = torch.qr(rand)
            return q
        end

        weights[6]:view(gru.hiddenSize,gru.hiddenSize):copy(ortho(gru.hiddenSize,gru.hiddenSize)) -- Ortho initialization
    end]]--

    local rnns = net:findModules('nn.SeqGRU_WN')
    for _,gru in pairs(rnns) do
        gru.bias:zero()

        local D, H = gru.inputSize, gru.outputSize
        
        local stdv = math.sqrt(1 / D) * math.sqrt(3) -- 'LeCunn' initialization
        gru.weight[{{1,D}}]:uniform(-stdv, stdv)
        
        stdv = math.sqrt(1 / H) * math.sqrt(3)
        gru.weight[{{D+1,D+H},{1,2*H}}]:uniform(-stdv, stdv)
        
        function ortho(inputDim,outputDim)
            local rand = torch.randn(outputDim,inputDim)
            local q,r = torch.qr(rand)
            return q
        end

        gru.weight[{{D+1,D+H},{2*H+1,3*H}}]:copy(ortho(H,H)) -- Ortho initialization
        gru:initFromWeight()
    end

    --[[for k,v in pairs(net:listModules()) do
        local type = torch.type(v)
        if type ~= 'nn.Sequential' and 
            type ~= 'nn.Bottle' and
            type ~= 'nn.ParallelTable' and
            type ~= 'nn.ConcatTable' then            
            local p = v:parameters()        
            if p and #p > 0 then
                print(type)
                print(p)
                if type == 'nn.SeqGRU_WN' then
                    local D,H = v.inputSize,v.outputSize
                    local W = v.v[{{1,D}}]
                    local G = v.g[{{1}}]
                    local B = v.bias

                    print("input.W", W:min(), W:max())
                    print("input.G", G:min(), G:max())
                    print("input.B", B:min(), B:max())

                    local W = v.v[{{D+1,D+H},{1,2*H}}]
                    local G = v.g[{{2},{1,2*H}}]

                    print("recurrent_gates.W", W:min(), W:max())
                    print("recurrent_gates.G", G:min(), G:max())

                    local W = v.v[{{D+1,D+H},{2*H+1,3*H}}]
                    local G = v.g[{{2},{2*H+1,3*H}}]

                    print("recurrent_candidate.W", W:min(), W:max())
                    print("recurrent_candidate.G", G:min(), G:max())
                else
                    for _k,_v in pairs(p) do
                        print(_v:min(),_v:max())
                    end                                
                end
            end
        end
    end]]--

    local gpus = torch.range(1, cutorch.getDeviceCount()):totable()
    net = nn.DataParallelTable(1,true,true):add(net,gpus):threads(function() -- TODO: optional nccl, optional GPU
        local cudnn = require 'cudnn'
        require 'rnn'
        require 'LinearWeightNorm'
        require 'SeqGRU_WN'
    end):cuda()

    return net
end

function get_files(path)    
    local files = {}
    for fname in paths.iterfiles(path) do
        table.insert(files, path..'/'..fname)        
    end

    return files
end

function create_thread_pool(n_threads)
    return threads.Threads(
        n_threads,
        function(threadId)
            local audio = require 'audio'
        end,
        function()
            function load(path)
                local aud = audio.load(path)                
                aud = aud:select(2,1) -- Mono only

                aud:csub(aud:min()) -- TODO: Choice of  'linear', 'mu-law', etc.
                aud:div(aud:max()+1e-16)                                

                local eps = 1e-5
                aud:mul(q_levels - eps)
                aud:add(eps / 2)
                aud:floor()
                aud:add(1)
                
                return aud
            end
        end
    )
end

function make_minibatch(thread_pool, files, indices, start, stop)
    local minibatch_size = stop - start + 1
    local dats = {}
    local dat = torch.Tensor(minibatch_size, seg_len)

    local j = 1
    for i = start,stop do
        local file_path = files[indices[i]]

        thread_pool:addjob(
            function(f)            
                return load(f)
            end,
            function(k)                
                dat[{j,{1,k:size(1)}}] = k                            
                j = j + 1                
            end,
            file_path
        )
    end
    
    thread_pool:synchronize()

    return dat
end

function resetStates(model)
    if model.impl then
        model.impl:exec(function(m)
            --local rnns = m:findModules('cudnn.GRU')--nn.SeqGRU')--
            local rnns = m:findModules('nn.SeqGRU_WN')--
            for i=1,#rnns do
                rnns[i]:forget()--resetStates()
                --rnns[i]:resetStates()
            end
        end)
    else
        --local rnns = model:findModules('cudnn.GRU')--nn.SeqGRU')--
        local rnns = model:findModules('nn.SeqGRU_WN')--
        for i=1,#rnns do
            rnns[i]:forget()--resetStates()
            --rnns[i]:resetStates()
        end
    end
end

function getSingleModel(model)
    return model.impl and model.impl:exec(function(model) return model end, 1)[1] or model
end

function train(net, files)
    net:training()

    local param,dparam = net:getParameters()
    --param:copy(torch.load("params.t7"))
    net:syncParameters()

    local criterion = nn.ClassNLLCriterion():cuda()

    --local optim_state = torch.load("optim_state.t7")
    local optim_state = {
        learningRate = learning_rate
    }

    local thread_pool = create_thread_pool(n_threads)

    local sample_idx = 1
    local total_iter = 0

    local losses = {}--torch.load("losses.t7")--{}
    local gradNorms = {}--torch.load("gradNorms.t7")--{}
    for epoch = 1,max_epoch do
        local total_err = 0

        local shuffled_files = torch.randperm(#files):long()

        local max_batches = math.floor(#files / minibatch_size)
        local n_batch = 0

        local start = 1
        while start <= #files do
            local stop = start + minibatch_size - 1
            if stop > #files then
                break
            end

            n_batch = n_batch + 1
            print("Mini-batch "..n_batch.."/"..max_batches)

            local minibatch = make_minibatch(thread_pool, files, shuffled_files, start, stop)
            local minibatch_seqs = minibatch:unfold(2,seq_len+big_frame_size,seq_len)
            
            local big_input_sequences = minibatch_seqs[{{},{},{1,-1-big_frame_size}}]
            local input_sequences = minibatch_seqs[{{},{},{big_frame_size-frame_size+1,-1-frame_size}}]
            local target_sequences = minibatch_seqs[{{},{},{big_frame_size+1,-1}}]
            local prev_samples = minibatch_seqs[{{},{},{big_frame_size-frame_size+1,-1-1}}]

            local big_frames = big_input_sequences:unfold(3,big_frame_size,big_frame_size)
            local frames = input_sequences:unfold(3,frame_size,frame_size)
            prev_samples = prev_samples:unfold(3,frame_size,1)
                        
            resetStates(net)
            for t=1,big_frames:size(2) do
                local start_time_tstep = sys.clock() 

                local _big_frames = big_frames:select(2,t):cuda()
                local _frames = frames:select(2,t):cuda()
                local _prev_samples = prev_samples:select(2,t):cuda()                
                
                local inp = {_big_frames,_frames,_prev_samples}
                local targets = target_sequences:select(2,t):cuda():view(-1)

                function feval(x)
                    if x ~= param then
                        param:copy(x)
                        net:syncParameters()
                    end

                    net:zeroGradParameters()
            
                    local output = net:forward(inp)
                    local flat_output = output:view(-1,q_levels)

                    local loss = criterion:forward(flat_output,targets)
                    local grad = criterion:backward(flat_output,targets)

                    net:backward(inp,grad)

                    dparam:clamp(-max_grad, max_grad)

                    local loss_bits = loss * math.log(math.exp(1),2) -- nats to bits
                    return loss_bits,dparam
                end  
                
                local _, err = optim.adam(feval,param,optim_state)

                local stop_time_tstep = sys.clock() 

                print(t.."/"..big_frames:size(2), err[1], grad_norm, stop_time_tstep - start_time_tstep)

                local grad_norm = dparam:norm(2)
                gradNorms[#gradNorms + 1] = grad_norm

                losses[#losses + 1] = err[1]

                total_err = total_err + err[1]
                total_iter = total_iter + 1
            end            

            local start_time = sys.clock() 

            --[[ TODO: Generate plots
            local lossesTensor = torch.Tensor(#losses)
            for i=1,#losses do
                lossesTensor[i] = losses[i]
            end
            ]]--

            print("Saving losses ...")
            torch.save("losses.t7", losses)                
            print("Saving gradNorms ...")
            torch.save("gradNorms.t7", gradNorms)                
            print("Saving optim state ...")
            torch.save("optim_state.t7", optim_state)                
            print("Saving params ...")
            torch.save("params.t7", param)
            print("Done!")

            local stop_time = sys.clock()

            print("Saved network and state (took "..(stop_time - start_time).." seconds)")
            
            start = stop + 1            
        end

        local err = total_err -- TODO: nats * math.log(math.exp(1),2) and avg over batch*timesteps
        print('Epoch: '..epoch..', loss = '..err)

        --TODO: Sample every
        --[[path.mkdir(string.format('samples/%d/',sample_idx))        
        sample(getSingleModel(net), string.format('samples/%d/sample.wav',sample_idx))
        sample_idx = sample_idx + 1]]--
    end
end

function sample(net,filepath)
    print("Sampling...")

    local big_frame_level_rnn = net:get(1):get(1)
    local frame_level_rnn = net:get(2):get(1):get(2)
    local sample_level_predictor = net:get(3)
    local big_rnn = big_frame_level_rnn:get(4)
    local frame_rnn = frame_level_rnn:get(3)

    net:evaluate()
    resetStates(net)

    local samples = torch.CudaTensor(n_samples, 1, sample_length)
    local big_frame_level_outputs, frame_level_outputs

    samples[{{},{},{1,big_frame_size}}] = q_zero -- Silence
    --samples[{{},{},{1,big_frame_size}}] = torch.floor(torch.rand(n_samples,1,big_frame_size)*q_levels) -- Uniform noise
    --samples[{{},{},{1,big_frame_size}}] = q_dat[{{2},{1},{1,big_frame_size}}]:expandAs(samples[{{},{},{1,big_frame_size}}]) -- A snippet of a sample from the training set

    --[[big_rnn.cellInput = torch.rand(1, n_samples, big_dim):cuda() - 0.5 -- Randomise the RNN initial state TODO: Fix, this doesn't work because you have to create, size and set the hiddenOutput tensors
    big_rnn.hiddenInput = torch.rand(1, n_samples, big_dim):cuda() - 0.5
    frame_rnn.cellInput = torch.rand(1, n_samples, big_dim):cuda() - 0.5
    frame_rnn.hiddenInput = torch.rand(1, n_samples, big_dim):cuda() - 0.5]]--

    local start_time = sys.clock()

    for t = big_frame_size + 1, sample_length do
        if (t-1) % big_frame_size == 0 then
            local big_frames = samples[{{},{},{t - big_frame_size, t - 1}}]
            big_frame_level_outputs = big_frame_level_rnn:forward(big_frames)
        end        

        if (t-1) % frame_size == 0 then
            local frames = samples[{{},{},{t - frame_size, t - 1}}]
            local _t = (((t-1) / frame_size) % (big_frame_size / frame_size)) + 1

            frame_level_outputs = frame_level_rnn:forward({big_frame_level_outputs[{{},{_t}}], frames})
        end

        local prev_samples = samples[{{},{},{t - frame_size, t - 1}}]
        
        local _t = (t-1) % frame_size + 1        

        local inp = {frame_level_outputs[{{},{_t}}], prev_samples}
        
        local sample = sample_level_predictor:forward(inp)        
        --sample:div(1.1) -- Sampling temperature TODO: make optional
        sample:exp()
        sample = torch.multinomial(sample:squeeze(),1)

        samples[{{},1,t}] = sample:typeAs(samples)
        
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

--[[local files = get_files(data_path)
local net = create_samplernn()
train(net,files)]]--

local net = create_samplernn()
local param,dparam=net:getParameters()
param:copy(torch.load("params.t7"))
path.mkdir("samples/1/")
sample(getSingleModel(net),'samples/1/sample.wav')