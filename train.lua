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

require 'nn'
require 'cunn'
require 'cudnn'
require 'rnn'
require 'optim'
require 'audio'
require 'xlua'
require 'SeqGRU_WN'
require 'SeqLSTM_WN'
require 'utils'

local threads = require 'threads'
threads.serialization('threads.sharedserialize')

local cmd = torch.CmdLine()
cmd:text('sampleRNN_torch: An Unconditional End-to-End Neural Audio Generation Model')
cmd:text('')

cmd:text('Session:')
cmd:option('-session','default','The name of the current training session')
cmd:option('-resume',false,'Resumes a previous training session')
cmd:text('')

cmd:text('Dataset:')
cmd:option('-dataset','','Specifies the training set to use')
cmd:text('')

cmd:text('GPU:')
cmd:option('-multigpu',true,'Enables multi-gpu support')
cmd:option('-use_nccl',true,'Enables NCCL support for DataParallelTable')
cmd:text('')

cmd:text('Sampling:')
cmd:option('-generate_samples',false,'If true will sample from the given model only (no training)')
cmd:option('-sample_every_epoch',true,'If true generates samples from the model every epoch')
cmd:option('-n_samples',5,'The number of samples to generate')
cmd:option('-sample_length',20,'The duration of generated samples')
cmd:option('-sampling_temperature',1,'The sampling temperature')
cmd:text('')

cmd:text('Model configuration:')
cmd:option('-cudnn_rnn',false,'Enables CUDNN for the RNN modules, when disabled a weight normalized version of SeqGRU is used')
cmd:option('-rnn_type','GRU','GRU | LSTM - Selects GRU or LSTM as the RNN type')
cmd:option('-q_levels',256,'The number of quantization levels')
cmd:option('-q_type','linear','linear | mu-law - The quantization scheme')
cmd:option('-norm_type','min-max','min-max | abs-max | none - The normalization scheme')
cmd:option('-embedding_size',256,'The dimension of the embedding vectors')
cmd:option('-big_frame_size',8,'The context size for the topmost tier RNN')
cmd:option('-frame_size',2,'The context size for the intermediate tier RNN')
cmd:option('-hidden_dim',1024,'The size of the hidden dimension')
cmd:option('-linear_type','WN','WN | default - Select weight normalized (WN) or standard (default) linear layers')
cmd:option('-dropout',false,'Enables dropout (only available for models using CUDNN)')
-- TODO: -learn_h0 -- Coming soon.
cmd:text('')

cmd:text('Training parameters:')
cmd:option('-learning_rate',0.001,'The learning rate to use')
cmd:option('-max_grad',1,'The per-dimension gradient clipping threshold')
cmd:option('-seq_len',512,'The number of TBPTT steps')
cmd:option('-minibatch_size',128,'Specifies the minibatch size to use')
cmd:option('-max_epoch',math.huge,'The maximum number of training epochs to perform')
cmd:text('')

local args = cmd:parse(arg)

local session_args = {'dataset','cudnn_rnn','rnn_type','q_levels','q_type','norm_type','embedding_size','big_frame_size','frame_size','hidden_dim','linear_type','dropout','learning_rate','max_grad','seq_len','minibatch_size'}

local session_path = 'sessions/'..args.session

if args.resume or args.generate_samples then
    local session = torch.load(session_path..'/session.t7')
    
    for k,v in pairs(session) do
        args[k] = v
    end
else    
    assert(args.session:len() > 0, 'session must be provided')
    assert(args.dataset:len() > 0, 'dataset must be provided')
    assert(args.linear_type == 'WN' or args.linear_type == 'default', 'linear_type must be "WN" or "default"')
    assert(args.q_type == 'mu-law' or args.q_type == 'linear', 'q_type must be "mu-law" or "linear"')
    assert(args.norm_type == 'min-max' or args.norm_type == 'abs-max' or args.norm_type == 'none', 'norm_type must be "min-max", "abs-max" or "none"')
    assert(args.rnn_type == 'GRU' or args.rnn_type == 'LSTM', 'rnn_type must be "GRU" or "LSTM"')

    path.mkdir('sessions/')
    path.mkdir(session_path)

    local session = {}
    for k,v in pairs(session_args) do
        session[v] = args[v]
    end

    torch.save(session_path..'/session.t7', session)
end

local audio_data_path = 'datasets/'..args.dataset..'/data'
local aud,sample_rate = audio.load(audio_data_path..'/p0001.wav')
local seg_len = aud:size(1)

local use_nccl = args.use_nccl
local multigpu = args.multigpu

local minibatch_size = args.minibatch_size
local n_threads = minibatch_size

local learning_rate = args.learning_rate
local max_grad = args.max_grad

local seq_len = args.seq_len

local linear_type = args.linear_type
local cudnn_rnn = args.cudnn_rnn
local rnn_type = args.rnn_type
local big_frame_size = args.big_frame_size
local frame_size = args.frame_size
local big_dim = args.hidden_dim
local dim = big_dim
local q_levels = args.q_levels
local q_zero = math.floor(q_levels / 2)
local q_type = args.q_type
local norm_type = args.norm_type
local emb_size = args.embedding_size
local dropout = args.dropout

local n_samples = args.n_samples
local sample_length = args.sample_length*sample_rate
local sampling_temperature = args.sampling_temperature

function create_samplernn()
    local big_rnn, frame_rnn
    if cudnn_rnn then
        big_rnn = cudnn[rnn_type](big_frame_size, big_dim, 1, true, dropout, true)
        frame_rnn = cudnn[rnn_type](dim, dim, 1, true, dropout, true)
    else 
        big_rnn = nn['Seq'..rnn_type..'_WN'](big_frame_size, big_dim)
        frame_rnn = nn['Seq'..rnn_type..'_WN'](dim, dim)

        big_rnn:remember('both')
        frame_rnn:remember('both')

        big_rnn.batchfirst = true
        frame_rnn.batchfirst = true
    end

    local linearType = linear_type == 'WN' and 'LinearWeightNorm' or 'Linear'
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

    if cudnn_rnn then
        if rnn_type == 'GRU' then
            local rnns = net:findModules('cudnn.GRU')
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
            end
        elseif rnn_type == 'LSTM' then
            local rnns = net:findModules('cudnn.LSTM')
            for _,lstm in pairs(rnns) do
                local biases = lstm:biases()[1]
                for k,v in pairs(biases) do
                    v:zero()
                end

                biases[2]:fill(3)

                local weights = lstm:weights()[1]

                local stdv = math.sqrt(1 / lstm.inputSize) * math.sqrt(3) -- 'LeCunn' initialization
                weights[1]:uniform(-stdv, stdv) 
                weights[2]:uniform(-stdv, stdv) 
                weights[3]:uniform(-stdv, stdv)
                weights[4]:uniform(-stdv, stdv)

                stdv = math.sqrt(1 / lstm.hiddenSize) * math.sqrt(3)
                weights[5]:uniform(-stdv, stdv) 
                weights[6]:uniform(-stdv, stdv) 
                weights[7]:uniform(-stdv, stdv)
                weights[8]:uniform(-stdv, stdv)
            end
        end
    else
        if rnn_type == 'GRU' then
            local rnns = net:findModules('nn.SeqGRU_WN')
            for _,gru in pairs(rnns) do
                local D, H = gru.inputSize, gru.outputSize

                gru.bias:zero()
                
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
        elseif rnn_type == 'LSTM' then
            local rnns = net:findModules('nn.Seq'..rnn_type..'_WN')
            for _,lstm in pairs(rnns) do
                local D, H = lstm.inputsize, lstm.outputsize

                lstm.bias:zero()
                lstm.bias[{{H + 1, 2 * H}}]:fill(3)
                
                local stdv = math.sqrt(1 / D) * math.sqrt(3) -- 'LeCunn' initialization
                lstm.weight[{{1,D}}]:uniform(-stdv, stdv)
                
                stdv = math.sqrt(1 / H) * math.sqrt(3)
                lstm.weight[{{D+1,D+H}}]:uniform(-stdv, stdv)                
                                
                lstm:initFromWeight()
            end
        end
    end

    if multigpu then
        local gpus = torch.range(1, cutorch.getDeviceCount()):totable()
        net = nn.DataParallelTable(1,true,use_nccl):add(net,gpus):threads(function()
            local cudnn = require 'cudnn'
            require 'rnn'
            require 'SeqGRU_WN'
            require 'SeqLSTM_WN'
        end):cuda()
    end

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
            require 'utils'
        end,
        function()
            function load(path)
                local aud = audio.load(path)   
                assert(aud:size(1) <= seg_len, 'Audio must be less than or equal to seg_len')
                assert(aud:size(2) == 1, 'Only mono training data is supported')
                aud = aud:view(-1)

                if norm_type == 'none' then
                    aud:csub(-0x80000000)
                    aud:div(0xFFFF0000)
                elseif norm_type == 'abs-max' then
                    aud:csub(-0x80000000)
                    aud:div(0xFFFF0000)
                    aud:mul(2)
                    aud:csub(1)
                    aud:div(math.max(math.abs(aud:min()),aud:max()))
                    aud:add(1)
                    aud:div(2)
                elseif norm_type == 'min-max' then
                    aud:csub(aud:min())
                    aud:div(aud:max())
                end

                if q_type == 'mu-law' then
                    aud:mul(2)
                    aud:csub(1)
                    aud = linear2mu(aud) + 1
                elseif q_type == 'linear' then
                    local eps = 1e-5
                    aud:mul(q_levels - eps)
                    aud:add(eps / 2)
                    aud:floor()
                    aud:add(1)
                end
                
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
            function(file_path)
                local aud = load(file_path)
                collectgarbage()

                return aud
            end,
            function(aud)
                dat[{j,{1,aud:size(1)}}] = aud
                j = j + 1                
            end,
            file_path
        )
    end
    
    thread_pool:synchronize()

    return dat
end

cudnn.RNN.forget = cudnn.RNN.resetStates

function resetStates(model)
    local rnn_lookup = cudnn_rnn and ('cudnn.'..rnn_type) or ('nn.Seq'..rnn_type..'_WN')
    if model.impl then
        model.impl:exec(function(m)
            local rnns = m:findModules(rnn_lookup)
            for i=1,#rnns do
                rnns[i]:forget()
            end
        end)
    else
        local rnns = model:findModules(rnn_lookup)
        for i=1,#rnns do
            rnns[i]:forget()
        end
    end
end

function getSingleModel(model)
    return model.impl and model.impl:exec(function(model) return model end, 1)[1] or model
end

function train(net, files)
    net:training()

    local criterion = nn.ClassNLLCriterion():cuda()

    local param,dparam = net:getParameters()
    if args.resume then param:copy(torch.load(session_path..'/params.t7')) end
    if multigpu then net:syncParameters() end

    local optim_state = args.resume and torch.load(session_path..'/optim_state.t7') or {learningRate = learning_rate}

    local losses = args.resume and torch.load(session_path..'/losses.t7') or {}    
    local gradNorms = args.resume and torch.load(session_path..'/gradNorms.t7') or {}

    local thread_pool = create_thread_pool(n_threads)
    
    local n_epoch = 0
    while n_epoch < args.max_epoch do
        local shuffled_files = torch.randperm(#files):long()
        local max_batches = math.floor(#files / minibatch_size)

        local epoch_err = 0
        local n_batch = 0
        local n_tbptt

        local start = 1
        while start <= #files do
            local stop = start + minibatch_size - 1
            if stop > #files then
                break
            end

            print('Mini-batch '..(n_batch + 1)..'/'..max_batches)

            local minibatch = make_minibatch(thread_pool, files, shuffled_files, start, stop)
            local minibatch_seqs = minibatch:unfold(2,seq_len+big_frame_size,seq_len)
            
            local big_input_sequences = minibatch_seqs[{{},{},{1,-1-big_frame_size}}]
            local input_sequences = minibatch_seqs[{{},{},{big_frame_size-frame_size+1,-1-frame_size}}]
            local target_sequences = minibatch_seqs[{{},{},{big_frame_size+1,-1}}]
            local prev_samples = minibatch_seqs[{{},{},{big_frame_size-frame_size+1,-1-1}}]

            local big_frames = big_input_sequences:unfold(3,big_frame_size,big_frame_size)
            local frames = input_sequences:unfold(3,frame_size,frame_size)
            prev_samples = prev_samples:unfold(3,frame_size,1)

            n_tbptt = big_frames:size(2)
                                    
            local batch_err = 0
            local minibatch_start_time = sys.clock()

            resetStates(net)
            for t=1,n_tbptt do
                local tstep_start_time = sys.clock() 

                local _big_frames = big_frames:select(2,t):cuda()
                local _frames = frames:select(2,t):cuda()
                local _prev_samples = prev_samples:select(2,t):cuda()                
                
                local inp = {_big_frames,_frames,_prev_samples}
                local targets = target_sequences:select(2,t):cuda():view(-1)

                function feval(x)
                    if x ~= param then
                        param:copy(x)
                        if multigpu then net:syncParameters() end
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

                local tstep_stop_time = sys.clock() 
                
                local grad_norm = dparam:norm(2)
                gradNorms[#gradNorms + 1] = grad_norm

                losses[#losses + 1] = err[1]

                epoch_err = epoch_err + err[1]
                batch_err = batch_err + err[1]

                local c = sys.COLORS
                print(string.format('%s%d%s/%s%d%s\tloss = %s%f%s grad_norm = %s%f%s time = %s%f%s seconds', 
                    c.cyan, t, 
                    c.white, c.cyan, n_tbptt,
                    c.white, c.cyan, err[1],
                    c.white, c.cyan, grad_norm,
                    c.white, c.cyan, tstep_stop_time - tstep_start_time,
                    c.white))
            end            

            local minibatch_stop_time = sys.clock()

            print('Minibatch: avg_loss = '..(batch_err / n_tbptt)..' time = '..(minibatch_stop_time - minibatch_start_time).. ' seconds')

            local save_start_time = sys.clock() 

            print('Saving losses ...')
            torch.save(session_path..'/losses.t7', losses)                

            print('Saving gradNorms ...')
            torch.save(session_path..'/gradNorms.t7', gradNorms)                

            print('Saving optim state ...')
            torch.save(session_path..'/optim_state.t7', optim_state)                

            print('Saving params ...')
            torch.save(session_path..'/params.t7', param)

            print('Done!')

            local save_stop_time = sys.clock()

            print('Saved network and state (took '..(save_stop_time - save_start_time)..' seconds)')
            
            start = stop + 1            
            n_batch = n_batch + 1
        end

        n_epoch = n_epoch + 1
        print('Epoch: '..n_epoch..', avg_loss = '..(epoch_err / (n_batch * n_tbptt)))

        if args.sample_every_epoch then
            sample(net, #losses)
        end
    end
end

function sample(net, n_iters)
    local parent_path = session_path..'/samples'
    path.mkdir(parent_path)

    local sample_path = parent_path..'/'..os.date('%H%M%S_%d%m%Y')..'_'..n_iters..'iters'
    path.mkdir(sample_path)

    generate_samples(getSingleModel(net), sample_path)    
end

function generate_samples(net,filepath)
    print('Sampling...')

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
    -- TODO: randomize initial state or use optional seed audio
    
    local sampling_start_time = sys.clock()

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
        sample:div(sampling_temperature)
        sample:exp()
        sample = torch.multinomial(sample:squeeze(),1)

        samples[{{},1,t}] = sample:typeAs(samples)
        
        xlua.progress(t-big_frame_size,sample_length-big_frame_size)
    end

    local sampling_stop_time = sys.clock()
    print('Generated '..(sample_length / sample_rate * n_samples)..' seconds of audio in '..(sampling_stop_time - sampling_start_time)..' seconds.')

    if q_type == 'mu-law' then
        samples = mu2linear(samples - 1)
        samples:add(1)
        samples:div(2)
    elseif q_type == 'linear' then
        samples = (samples - 1) / (q_levels - 1)
    end

    local audioOut = -0x80000000 + 0xFFFF0000 * samples
    for i=1,audioOut:size(1) do
        audio.save(filepath..'/'..string.format('%d.wav',i), audioOut:select(1,i):t():double(), sample_rate)
    end

    print('Audio saved.')

    net:training()
end

local net = create_samplernn()

if args.generate_samples then
    local param,dparam = net:getParameters()
    param:copy(torch.load(session_path..'/params.t7'))

    local n_iters = #torch.load(session_path..'/losses.t7')

    sample(net, n_iters)
else
    local files = get_files(audio_data_path)
    train(net,files)
end