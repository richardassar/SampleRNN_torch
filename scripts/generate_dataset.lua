require 'audio'
require 'xlua'

local cmd = torch.CmdLine()
cmd:text('generate_dataset.lua options:')
cmd:option('-source_path','','The path containing source audio')
cmd:option('-dest_path','','Where to store the audio segments')
cmd:option('-seg_len',8,'The length in seconds of each audio segment')

local args = cmd:parse(arg)
assert(args.source_path:len() > 0, "source_path must be provided")
assert(args.dest_path:len() > 0, "dest_path must be provided")
assert(args.seg_len > 0, "seg_len must be positive")

function get_files(path)    
	local files = {}
	for fname in paths.iterfiles(path) do
		table.insert(files, path..'/'..fname)        
	end

	return files
end

print("Generating training set from '"..args.source_path.."'")
local files = get_files(args.source_path)

local idx = 1
local sample_rate_check
for i,filepath in pairs(files) do
	print("Processing "..i.."/"..#files..":")

	local aud,sample_rate = audio.load(filepath)
	assert(sample_rate_check == nil or sample_rate_check == sample_rate, "Sample rate mismatch")
	sample_rate_check = sample_rate

	aud = aud:sum(2) -- Mix stereo channels
	aud = aud:view(-1)
	aud:csub(aud:mean()) -- Remove DC component
	aud:div(math.max(math.abs(aud:min()),aud:max())) -- Normalize to abs-max amplitude
	aud:add(1) -- Scale to [0,1]
	aud:div(2)

	local seglen_samples = args.seg_len * sample_rate
	local n_segs = math.floor(aud:size(1)/seglen_samples)
	for i=1,n_segs do
		local aud = aud:narrow(1,(i-1)*seglen_samples+1,seglen_samples):view(-1,1)
		if aud:min() ~= aud:max() then -- skip silence
			aud = -0x80000000 + 0xFFFF0000 * aud
			audio.save(string.format("%s/p%04d.wav",args.dest_path,idx), aud, sample_rate)
			idx = idx + 1
		end

		xlua.progress(i,n_segs)
	end
end

print("Done!")