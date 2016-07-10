require 'mattorch'
require 'cunn'
require 'nn'
require 'optim'
require 'image'
require 'xlua'
local c = require 'trepl.colorize'
opt = lapp[[
   -b,--batchSize             (default 100)          batch size
   -r,--learningRate          (default 0.2)        learning rate
   --learningRateDecay        (default 1e-7)      learning rate decay
   --weightDecay              (default 0.0005)      weightDecay
   -m,--momentum              (default 0.9)         momentum
   --epoch_step               (default 3)          epoch step
   --max_epoch                (default 36)           maximum number of iteration
    --model                    (default ConvNet)     model name
]]
--6-class classification problem
classes = { 'barren_land', 'trees', 'grassland', 'roads', 'buildings', 'water_bodies'}

--load model
model=dofile('models/'..opt.model..'.lua')

--transfer model to GPU
model:cuda()

-- retrieve parameters and gradients
parameters,gradParameters = model:getParameters()

--print model
print('<SAT-6 dataset> using model:')
print(model)

--setting criterion
print('==>' ..' setting criterion')
criterion = nn.CrossEntropyCriterion():cuda()

--set dimensions of confusion matrix
confusion = optim.ConfusionMatrix(#classes)

--define number of patches for training and testing
trsize=324000
tesize=81000

--load dataset
print '==> loading dataset'
--local matio = require 'matio'
dataset=mattorch.load('sat-6-full.mat')

--adjust dimensions so that they are appropriate for torch
trd1=dataset.train_x

trl1=dataset.train_y

ted1=dataset.test_x

tel1=dataset.test_y

trainData = {
   data = trd1:double(),
   labels = trl1:double(),
   size = function() return trsize end
}

testData = {
   data = ted1:double(),
   labels = tel1:double(),
   size = function() return trsize end
}

--extract the desirable number of training and testing patches
trainData.data=trainData.data[{{1,trsize}}]
trainData.labels=trainData.labels[{{1,trsize}}]

testData.data=testData.data[{{1,tesize}}]
testData.labels=testData.labels[{{1,tesize}}]

--label editing
ftrl=torch.Tensor(trsize,1)
ftel=torch.Tensor(tesize,1)

for i=1,trsize do
 for j=1,6 do
  if trainData.labels[{ i,j }]==1 then
  ftrl[i]=j
  end
 end
end

trainData.labels=ftrl

for i=1,tesize do
 for j=1,6 do
  if testData.labels[{ i,j }]==1 then
  ftel[i]=j
  end
 end
end

testData.labels=ftel

print('Size of training patches :')
print(trainData.data:size())
print('Size of testing patches :')
print(testData.data:size())

--preprocess/normalize train/test sets
print '<trainer> preprocessing data (color space + normalization)'
collectgarbage()

mean = {} -- store the mean, to normalize the test set in the future
stdv  = {} -- store the standard-deviation for the future
for i=1,4 do -- over each image channel
    mean[i] = trainData.data[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
    print('Channel ' .. i .. ', Mean: ' .. mean[i])
    trainData.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
    
    stdv[i] = trainData.data[{ {}, {i}, {}, {}  }]:std() -- std estimation
    print('Channel ' .. i .. ', Standard Deviation: ' .. stdv[i])
    trainData.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end


for i=1,4 do -- over each image channel
    testData.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction    
    testData.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end

--saving mean and stdv
torch.save('mean_sat6.t7',mean)
torch.save('stdv_sat6.t7',stdv)

--configuring optimizer
print(c.blue'==>' ..' configuring optimizer')
optimState = {
  learningRate = opt.learningRate,
  weightDecay = opt.weightDecay,
  momentum = opt.momentum,
  learningRateDecay = opt.learningRateDecay,
}

-- training function
function train()
  model:training()
  epoch = epoch or 1

  -- drop learning rate every "epoch_step" epochs
 if epoch % opt.epoch_step == 0 then optimState.learningRate = optimState.learningRate/2 end
  
  print(c.blue '==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

  local targets = torch.CudaTensor(opt.batchSize)
  local indices = torch.randperm(trainData.data:size(1)):long():split(opt.batchSize)

  -- remove last element so that all the batches have equal size
  indices[#indices] = nil

  local tic = torch.tic()
  for t,v in ipairs(indices) do
    xlua.progress(t, #indices)

    local inputs = trainData.data:index(1,v)
    targets:copy(trainData.labels:index(1,v))

    local feval = function(x)
      if x ~= parameters then parameters:copy(x) end
      gradParameters:zero()
      
      local outputs = model:forward(inputs)
      local f = criterion:forward(outputs, targets)
      local df_do = criterion:backward(outputs, targets)
      model:backward(inputs, df_do)
      confusion:batchAdd(outputs, targets)

      return f,gradParameters
    end
    optim.sgd(feval, parameters, optimState)
end   

  confusion:updateValids()
  print(('Train accuracy: '..c.cyan'%.2f'..' %%\t time: %.2f s'):format(
        confusion.totalValid * 100, torch.toc(tic)))

  train_acc = confusion.totalValid * 100
  print(confusion)
  print('Train Accuracy:'..train_acc)
  confusion:zero()

  --saving model
  torch.save('model_sat6_3.net',model)
  epoch = epoch + 1
end

--test function
function test()

-- disable flips, dropouts and batch normalization
  model:evaluate()
  print('==>'.." testing")
  local bs = 125
  klaseis=torch.DoubleTensor(tesize,1)
  for i=1,testData.data:size(1),bs do
     outputs = model:forward(testData.data:narrow(1,i,bs))
     confusion:batchAdd(outputs, testData.labels:narrow(1,i,bs))
     idx=torch.DoubleTensor(125,1):fill(0)

     --extract class predictions and put it to Tensor 'klaseis'
      for v=1,125 do
      meg=outputs[v]:max()
      for u=1,6 do
         if outputs[v][u]==meg then idx[v]=u end 
      end
    end       
    klaseis[{{i,i+124}}]=idx
  end

  confusion:updateValids()
  print('Test accuracy:', confusion.totalValid * 100)
  print(confusion)
  confusion:zero()
end


for i=1,opt.max_epoch do
  train()
  test()

--writing test predictions of the trained model to file
if i==36 then
torch.save('class_predictions_sat6.t7',klaseis)
end
end

  
  
