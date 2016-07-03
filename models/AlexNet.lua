require 'nn'
local SpatialConvolution = nn.SpatialConvolution--lib[1]
local SpatialMaxPooling = nn.SpatialMaxPooling--lib[2]

local features = nn.Sequential()
features:add(SpatialConvolution(4,16,3,3,1,1,0,0))       -- 28 -> 26
features:add(nn.ReLU(true))
features:add(SpatialMaxPooling(2,2,2,2))                 -- 26 -> 13
features:add(SpatialConvolution(16,48,3,3,1,1,1,1))      -- 13 -> 13
features:add(nn.ReLU(true))
features:add(SpatialMaxPooling(3,3,2,2))                 -- 13 -> 6
features:add(SpatialConvolution(48,96,3,3,1,1,1,1))      -- 6 -> 6
features:add(nn.ReLU(true))
features:add(SpatialConvolution(96,64,3,3,1,1,1,1))      -- 6 -> 6
features:add(nn.ReLU(true))
features:add(SpatialConvolution(64,64,3,3,1,1,1,1))      -- 6 -> 6
features:add(nn.ReLU(true))
features:add(SpatialMaxPooling(2,2,2,2))                 -- 6 -> 3

local classifier = nn.Sequential()
classifier:add(nn.View(64*3*3))                          -- 576
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(64*3*3, 200))                   -- 576 -> 200
classifier:add(nn.Threshold(0, 1e-6))
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(200, 200))                      -- 200 -> 200
classifier:add(nn.Threshold(0, 1e-6))
classifier:add(nn.Linear(200, #classes))                 -- 200 -> 4
classifier:add(nn.LogSoftMax())

local model = nn.Sequential()
model:add(nn.Copy('torch.DoubleTensor','torch.CudaTensor'))
model:add(features):add(classifier)

return model
