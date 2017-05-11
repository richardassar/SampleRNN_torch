require 'audio'
require 'gnuplot'

local cmd = torch.CmdLine()
cmd:text('generate_plots.lua - plots the loss and gradNorm curve for a given session')
cmd:text('')

cmd:text('Session:')
cmd:option('-session','default','The name of the session for which to generate plots')
cmd:text('')

local args = cmd:parse(arg)
local session_path = 'sessions/'..args.session

path.mkdir(session_path..'/plots')

local session = torch.load(session_path..'/session.t7')
local losses = torch.load(session_path..'/losses.t7')
local grads = torch.load(session_path..'/gradNorms.t7')

local audio_data_path = 'datasets/'..session.dataset..'/data'
local aud,sample_rate = audio.load(audio_data_path..'/p0001.wav')
local n_tsteps = math.floor((aud:size(1) - session.big_frame_size) / session.seq_len)

print(#losses..' iterations')

local lossesTensor = torch.Tensor(#losses)
for i=1,#losses do
    lossesTensor[i] = losses[i]
end

local gradsTensor = torch.Tensor(#grads)
for i=1,#grads do
    gradsTensor[i] = grads[i]
end

print('Plotting loss curve ...')

local loss_max = lossesTensor:view(-1,n_tsteps):max(2)
lossesTensor:clamp(0,lossesTensor:view(-1,n_tsteps):max(2)[{{2,-1}}]:max())

loss_max = lossesTensor:view(-1,n_tsteps):max(2)
local loss_min = lossesTensor:view(-1,n_tsteps):min(2)
local loss_mean = lossesTensor:view(-1,n_tsteps):mean(2)

gnuplot.pdffigure(session_path..'/plots/loss_curve.pdf')
gnuplot.raw('set size rectangle')
gnuplot.raw('set xlabel "minibatches"') 
gnuplot.raw('set ylabel "NLL (bits)"') 
gnuplot.plot({'min',loss_min,'-'},{'max',loss_max,'-'},{'mean',loss_mean,'-'})
gnuplot.plotflush()
gnuplot.close()

print('Plotting grad curve ...')

gnuplot.pdffigure(session_path..'/plots/grad_curve.pdf')
gnuplot.raw('set size rectangle')
gnuplot.raw('set xlabel "iterations"') 
gnuplot.raw('set ylabel "norm(dparam)"') 
gnuplot.plot({gradsTensor,'-'})
gnuplot.plotflush()
gnuplot.close()

print('Done!')