#%% base dir
def getBase():
    # imports
    import os
    import sys
    
    # declation (start)
    print('** Getting the Base folder **')
    
    # base folder
    folderBase = os.path.dirname(os.path.realpath(sys.argv[0])).replace('\\', '/')+'/'

    # declation (end)
    print('done.')
    print('')

    # return
    return folderBase

#%% installation
def installReq(folderBase):
    # imports
    import subprocess
    
    # declation (start)
    print('** Installing the Requirements **')

    # installation
    file = open(folderBase+'dsl_data/metadata/base.txt', 'w+')
    file.write(folderBase)
    modules = open(folderBase+'/requirements/requirements.txt').read().split(' ')
    for module in modules:
        try:
            __import__(module)
        except ImportError:
            subprocess.run(['pip', 'install', module], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    
    # declation (end)
    print('done.')
    print('')

#%% preparation
def preparation(folderBase):
    # imports
    import os
    import shutil
    import sys
    import json
    import pandas as pd
    
    # declation (start)
    print('** Preparation **')
    
    # checking files
    folderData = folderBase+'/dsl_data/'
    if not os.path.exists(folderData):
        sys.exit('"dsl_data" directory does not exist!')
    
    # preparing datasets
    datasetDevOld = pd.read_csv(folderData+'development.csv')
    datasetEvalOld = pd.read_csv(folderData+'evaluation.csv')
    
    datasetDev = pd.DataFrame({
        'path': datasetDevOld['path'],
        'gender': datasetDevOld['gender'],
        'langFirst': datasetDevOld['First Language spoken'],
        'langSecond': datasetDevOld['Current language used for work/school'],
        'age': datasetDevOld['ageRange'],
        'fluency': datasetDevOld['Self-reported fluency level '],
        'action': datasetDevOld['action'],
        'object': datasetDevOld['object'],
        'intent': datasetDevOld['action']+datasetDevOld['object']
        })
    
    datasetEval = pd.DataFrame({
        'path': datasetEvalOld['path'],
        'gender': datasetEvalOld['gender'],
        'langFirst': datasetEvalOld['First Language spoken'],
        'langSecond': datasetEvalOld['Current language used for work/school'],
        'age': datasetEvalOld['ageRange'],
        'fluency': datasetEvalOld['Self-reported fluency level ']
        })
    
    # storing datasets
    if not os.path.exists(folderData+'datasets/'):
        os.makedirs(folderData+'datasets/')
    datasetDev.to_csv(folderData+'datasets/development.csv', index=False)
    datasetEval.to_csv(folderData+'datasets/evaluation.csv', index=False)
    
    # preparing metadata
    labels = {}
    for label in ['gender', 'langFirst', 'langSecond', 'age', 'fluency', 'action', 'object', 'intent']:
        temp = datasetDev[label].value_counts()
        labels[label] = {
            'size': len(temp),
            'values': list(temp.index),
            'counts': [int(value) for value in temp.values]
            }
    
    # storing metadata
    if not os.path.exists(folderData+'metadata/config.json'):
        shutil.copy(folderData+'metadata/configDefault.json', folderData+'metadata/config.json')
    if not os.path.exists(folderData+'metadata/'):
        os.makedirs(folderData+'metadata/')
    with open(folderData+'metadata/labels.json', 'w') as file:
        json.dump(labels, file, indent=4)
        
    # declation (end)
    print('done.')
    print('')

#%% preprocess of development dataset
def preprocessDev(folderBase):
    # imports
    import os
    import sys
    import json
    import pandas as pd
    import shutil
    import pickle
    import numpy as np
    import iteround
    import torch
    import tqdm
    import extraClasses
    
    # checking files
    folderData = folderBase+'/dsl_data/'
    if not os.path.exists(folderData):
        sys.exit('"dsl_data" directory does not exist!')
    directories = ['datasets', 'metadata', 'audio']
    for directory in directories:
        if not os.path.exists(folderData+directory):
            sys.exit('"'+directory+'" directory does not exist!')
    
    # loading dataset
    dataset = pd.read_csv(folderData+'datasets/development.csv')
    paths = folderBase+dataset['path']
    indexes = dataset.index
    sizeData = dataset.shape[0]
    
    # loading metadata
    with open(folderData+'metadata/labels.json', 'r') as file:
        labels = json.load(file)
    if os.path.exists(folderData+'metadata/config.json'):
        with open(folderData+'metadata/config.json', 'r') as file:
            config = json.load(file)
    else:
        with open(folderData+'metadata/configDefault.json', 'r') as file:
            config = json.load(file)
    
    # preparing the storage
    folderBatches = folderData+'batches/'
    if os.path.exists(folderBatches):
        shutil.rmtree(folderBatches)
    os.makedirs(folderBatches)
    
    # getting classes
    labelTransformer = extraClasses.LabelTransformer(folderBase)
    reader = extraClasses.Reader(folderBase)
    augmenter = extraClasses.Augmenter(folderBase)
    aligner = extraClasses.Aligner(folderBase)
    featureExtractor = extraClasses.FeatureExtractor(folderBase)
    
    # reading audio files
    waveforms = []
    lengths = []
    references = []
    isAugmenteds = []
    print('** Reading Audio Files from Development Dataset **')
    for i in tqdm.tqdm(range(sizeData)):
        waveform = reader(paths[i])
        waveforms.append(waveform)
        lengths.append(waveform.shape[0])
        references.append(indexes[i])
        isAugmenteds.append(0)
    
    # augmentation
    mapper = pd.Series(
        index = labels['intent']['values'],
        data = 1/np.array(labels['intent']['counts'])
        )
    coeffs = mapper[dataset['intent']]
    coeffs = coeffs*sizeData*(config['preprocessing']['augmenter']['nTriesAvg'])/coeffs.sum()
    coeffs = config['preprocessing']['augmenter']['proportionality']*coeffs
    coeffs += (1-config['preprocessing']['augmenter']['proportionality'])*np.full(sizeData, config['preprocessing']['augmenter']['nTriesAvg'])
    coeffs = iteround.saferound(coeffs, 0)
    coeffs = list(map(round, coeffs))
    sizeDataAug = 0
    waveformsAug = []
    lengthsAug = []
    referencesAug = []
    isAugmentedsAug = []
    print('** Augmenting Audio Files in Development Dataset **')
    for i in tqdm.tqdm(range(sizeData)):
        nTries = coeffs[i]
        waveformsNew, lengthsNew = augmenter(waveforms[i], nTries)
        sizeDataAug += nTries
        waveformsAug += waveformsNew
        lengthsAug += lengthsNew
        referencesAug += [i]*nTries
        isAugmentedsAug += [1]*nTries
    sizeData += sizeDataAug
    waveforms += waveformsAug
    lengths += lengthsAug
    references += referencesAug
    isAugmenteds += isAugmentedsAug
    
    # batching, alignment, and feature extraction
    pattern = np.random.permutation(sizeData).tolist()
    batchesStarts = list(range(0, sizeData, config['preprocessing']['batcher']['size']))
    batchesEnds = batchesStarts[1:]+[sizeData]
    batchesIntervals = [[batchesStarts[i], batchesEnds[i]] for i in range(len(batchesStarts))]
    batches = {}
    print('** Batching, Alignment, and Feature Extraction of Development Dataset **')
    for b, batchInterval in enumerate(tqdm.tqdm(batchesIntervals)):
        sizeBatch = batchInterval[1] - batchInterval[0]
        indexesBatch = pattern[batchInterval[0]:batchInterval[1]]
    
        # batching
        waveformsBatch = []
        lengthsBatch = []
        referencesBatch = []
        isAugmentedsBatch = []
        for i in indexesBatch:
            waveformsBatch.append(waveforms[i])
            lengthsBatch.append(lengths[i])
            referencesBatch.append(references[i])
            isAugmentedsBatch.append(isAugmenteds[i])
    
        # alignment
        lengthBatch = int(np.quantile(lengthsBatch, config['preprocessing']['aligner']['quantile']))
        for i in range(sizeBatch):
            waveformsBatch[i] = aligner(waveformsBatch[i], lengthBatch)
        waveformsBatch = torch.from_numpy(np.array(waveformsBatch, dtype=np.float32))
    
        # feature extraction
        XBatch = featureExtractor(waveformsBatch)
        if XBatch.isnan().sum() != 0 or XBatch.isinf().sum() != 0:
            freaks = torch.hstack([XBatch.isnan().nonzero()[:, 0], XBatch.isinf().nonzero()[:, 0]]).unique()
            XBatch[freaks, :] = 0
        (yGenderBatch,
            yLangFirstBatch,
            yLangSecondBatch,
            yAgeBatch,
            yFluencyBatch,
            yActionBatch,
            yObjectBatch,
            yBatch) = labelTransformer.labelEncode(dataset.drop(columns=['path']).loc[referencesBatch, :])
        
        # storage
        batchName = f'batchDev{b:04d}'
        batches[batchName] = {
            'size': sizeBatch,
            'length': lengthBatch,
            'references': referencesBatch,
            'isAugmenteds': isAugmentedsBatch
            }
        with open(folderBatches+batchName+'.pickle', 'wb') as file:
            pickle.dump(
                obj = (XBatch, yGenderBatch, yLangFirstBatch, yLangSecondBatch, yAgeBatch, yFluencyBatch, yActionBatch, yObjectBatch, yBatch),
                file = file,
                protocol = pickle.HIGHEST_PROTOCOL)
    with open(folderData+'metadata/batchesDev.json', 'w') as file:
        json.dump(batches, file, indent=4)
        
    # declation (end)
    print('')

#%% preprocess of evaluation dataset
def preprocessEval(folderBase):
    # imports
    import os
    import sys
    import json
    import pandas as pd
    import pickle
    import numpy as np
    import torch
    import tqdm
    import extraClasses
    
    # checking files
    folderData = folderBase+'/dsl_data/'
    if not os.path.exists(folderData):
        sys.exit('"dsl_data" directory does not exist!')
    directories = ['datasets', 'metadata', 'audio']
    for directory in directories:
        if not os.path.exists(folderData+directory):
            sys.exit('"'+directory+'" directory does not exist!')
    
    # loading dataset
    dataset = pd.read_csv(folderData+'datasets/evaluation.csv')
    paths = folderBase+dataset['path']
    indexes = dataset.index
    sizeData = dataset.shape[0]
    
    # loading metadata
    if os.path.exists(folderData+'metadata/config.json'):
        with open(folderData+'metadata/config.json', 'r') as file:
            config = json.load(file)
    else:
        with open(folderData+'metadata/configDefault.json', 'r') as file:
            config = json.load(file)
    
    # getting classes
    labelTransformer = extraClasses.LabelTransformer(folderBase)
    reader = extraClasses.Reader(folderBase)
    aligner = extraClasses.Aligner(folderBase)
    featureExtractor = extraClasses.FeatureExtractor(folderBase)
    
    # reading audio files
    waveforms = []
    lengths = []
    references = []
    isAugmenteds = []
    print('** Reading Audio Files from Evaluation Dataset **')
    for i in tqdm.tqdm(range(sizeData)):
        waveform = reader(paths[i])
        waveforms.append(waveform)
        lengths.append(waveform.shape[0])
        references.append(indexes[i])
        isAugmenteds.append(0)
    
    # batching, alignment, and feature extraction
    folderBatches = folderData+'batches/'
    pattern = np.random.permutation(sizeData).tolist()
    batchesStarts = list(range(0, sizeData, config['preprocessing']['batcher']['size']))
    batchesEnds = batchesStarts[1:]+[sizeData]
    batchesIntervals = [[batchesStarts[i], batchesEnds[i]] for i in range(len(batchesStarts))]
    batches = {}
    print('** Batching, Alignment, and Feature Extraction of Evaluation Dataset **')
    for b, batchInterval in enumerate(tqdm.tqdm(batchesIntervals)):
        sizeBatch = batchInterval[1] - batchInterval[0]
        indexesBatch = pattern[batchInterval[0]:batchInterval[1]]
    
        # batching
        waveformsBatch = []
        lengthsBatch = []
        referencesBatch = []
        for i in indexesBatch:
            waveformsBatch.append(waveforms[i])
            lengthsBatch.append(lengths[i])
            referencesBatch.append(references[i])
    
        # alignment
        lengthBatch = int(np.quantile(lengthsBatch, config['preprocessing']['aligner']['quantile']))
        for i in range(sizeBatch):
            waveformsBatch[i] = aligner(waveformsBatch[i], lengthBatch)
        waveformsBatch = torch.from_numpy(np.array(waveformsBatch, dtype=np.float32))
    
        # feature extraction
        XBatch = featureExtractor(waveformsBatch)
        if XBatch.isnan().sum() != 0 or XBatch.isinf().sum() != 0:
            freaks = torch.hstack([XBatch.isnan().nonzero()[:, 0], XBatch.isinf().nonzero()[:, 0]]).unique()
            XBatch[freaks, :] = 0
        (yGenderBatch,
            yLangFirstBatch,
            yLangSecondBatch,
            yAgeBatch,
            yFluencyBatch) = labelTransformer.labelEncode(dataset.drop(columns=['path']).loc[referencesBatch, :])
        
        # storage
        batchName = f'batchEval{b:04d}'
        batches[batchName] = {
            'size': sizeBatch,
            'length': lengthBatch,
            'references': referencesBatch
            }
        with open(folderBatches+batchName+'.pickle', 'wb') as file:
            pickle.dump(
                obj = (XBatch, yGenderBatch, yLangFirstBatch, yLangSecondBatch, yAgeBatch, yFluencyBatch),
                file = file,
                protocol = pickle.HIGHEST_PROTOCOL)
    with open(folderData+'metadata/batchesEval.json', 'w') as file:
        json.dump(batches, file, indent=4)
        
    # declation (end)
    print('')
    
#%% development
def develop(folderBase):
    # imports
    import sys
    import json
    import os
    import random
    import pickle
    import pandas as pd
    import torch
    import torch.nn as nn
    import tqdm
    import time
    import extraClasses
    
    # checking files
    folderData = folderBase+'/dsl_data/'
    if not os.path.exists(folderData):
        sys.exit('"dsl_data" directory does not exist!')
    directories = ['metadata', 'batches']
    for directory in directories:
        if not os.path.exists(folderData+directory):
            sys.exit('"'+directory+'" directory does not exist!')
    
    # loading metadata
    if os.path.exists(folderData+'metadata/config.json'):
        with open(folderData+'metadata/config.json', 'r') as file:
            config = json.load(file)
    else:
        with open(folderData+'metadata/configDefault.json', 'r') as file:
            config = json.load(file)
    with open(folderData+'metadata/batchesDev.json', 'r') as file:
        batches = json.load(file)
    batchNames = list(batches.keys())
    if config['network']['general']['splitter']['doShuffle']:
        random.shuffle(batchNames)
    
    # loading batches
    folderBatches = folderData+'batches/'
    XBatches = {}
    yGenderBatches = {}
    yLangFirstBatches = {}
    yLangSecondBatches = {}
    yAgeBatches = {}
    yFluencyBatches = {}
    yActionBatches = {}
    yObjectBatches = {}
    yBatches = {}
    print('** Loading Development Batches **')
    for batchName in tqdm.tqdm(batchNames):
        with open(folderBatches+batchName+'.pickle', 'rb') as file:
            XBatch, yGenderBatch, yLangFirstBatch, yLangSecondBatch, yAgeBatch, yFluencyBatch, yActionBatch, yObjectBatch, yBatch = pickle.load(file)
            XBatches[batchName] = XBatch
            yGenderBatches[batchName] = yGenderBatch
            yLangFirstBatches[batchName] = yLangFirstBatch
            yLangSecondBatches[batchName] = yLangSecondBatch
            yAgeBatches[batchName] = yAgeBatch
            yFluencyBatches[batchName] = yFluencyBatch
            yActionBatches[batchName] = yActionBatch
            yObjectBatches[batchName] = yObjectBatch
            yBatches[batchName] = yBatch
    
    # splitting data
    position = int(config['network']['general']['splitter']['ratioTrain']*len(batchNames))
    batchNamesTrain = batchNames[:position]
    batchNamesTest = batchNames[position:]
    
    # development
    # device
    if config['network']['general']['useCuda']:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'
    
    # network
    network = extraClasses.Network(folderBase, device).to(device)
    
    # optimizer
    optimizer = torch.optim.Adam(
        network.parameters(),
        lr = config['network']['general']['learningRate'],
        weight_decay = config['network']['general']['penaltyL2'])
    
    # scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer = optimizer,
        milestones = config['network']['general']['scheduler']['milestones'],
        gamma = config['network']['general']['scheduler']['rateDecay'])
    
    # criterion
    criterion = nn.CrossEntropyLoss()
    
    # development
    statistics = pd.DataFrame(
        columns = ['epoch', 'accTrain', 'accTest', 'accTestClean', 'duration']
    )
    for e in range(config['network']['general']['nEpochs']):
        # training
        timeStart = time.perf_counter()
        successes = 0
        total = 0
        network.train()
        print(f'** Epoch {e} **')
        bar = tqdm.tqdm(batchNamesTrain, desc='loss: inf | train accuracy: 0')
        for batchName in bar:
            # batches
            XBatch = XBatches[batchName].to(device)
            yGenderBatch = yGenderBatches[batchName].to(device)
            yLangFirstBatch = yLangFirstBatches[batchName].to(device)
            yLangSecondBatch = yLangSecondBatches[batchName].to(device)
            yAgeBatch = yAgeBatches[batchName].to(device)
            yFluencyBatch = yFluencyBatches[batchName].to(device)
            yActionBatch = yActionBatches[batchName].to(device)
            yObjectBatch = yActionBatches[batchName].to(device)
            yBatch = yBatches[batchName].to(device)
    
            # forward pass
            outActionBatch, outObjectBatch, outBatch = network(
                X = XBatch,
                yGender = yGenderBatch,
                yLangFirst = yLangFirstBatch,
                yLangSecond = yLangSecondBatch,
                yAge = yAgeBatch,
                yFluency = yFluencyBatch,
                yAction = yActionBatch,
                epoch = e
                )
            
            # loss calculation
            loss = criterion(outActionBatch, yActionBatch) + criterion(outBatch, yBatch)
            # loss = criterion(outActionBatch, yActionBatch) + criterion(outObjectBatch, yObjectBatch) + criterion(outBatch, yBatch)
            
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # optimization
            optimizer.step()
            
            # prediction
            predBatch = network.predictor(outBatch)
            
            # summations
            successes += (predBatch.max(1)[1] == yBatch.nonzero()[:, 1]).sum()
            total += XBatch.shape[0]
    
            # logging
            accTrain = successes/total
            bar.set_description(f'loss: {loss:.4f} | train accuracy: {100*accTrain:0.2f}% | Progress')
        
        #testing
        network.eval()
        with torch.no_grad():
            successes = 0
            total = 0
            successesClean = 0
            totalClean = 0
            for batchName in batchNamesTest:
            # batches
                XBatch = XBatches[batchName].to(device)
                yGenderBatch = yGenderBatches[batchName].to(device)
                yLangFirstBatch = yLangFirstBatches[batchName].to(device)
                yLangSecondBatch = yLangSecondBatches[batchName].to(device)
                yAgeBatch = yAgeBatches[batchName].to(device)
                yFluencyBatch = yFluencyBatches[batchName].to(device)
                yActionBatch = yActionBatches[batchName].to(device)
                yObjectBatch = yActionBatches[batchName].to(device)
                yBatch = yBatches[batchName].to(device)
    
                # outputs
                _, _, outBatch = network(
                    X = XBatch,
                    yGender = yGenderBatch,
                    yLangFirst = yLangFirstBatch,
                    yLangSecond = yLangSecondBatch,
                    yAge = yAgeBatch,
                    yFluency = yFluencyBatch
                    )
                
                # prediction
                predBatch = network.predictor(outBatch)
                
                # summations
                isReals = 1 - torch.tensor(batches[batchName]['isAugmenteds']).to(device)
                successes += (predBatch.max(1)[1] == yBatch.nonzero()[:, 1]).sum()
                total += XBatch.shape[0]
                successesClean += ((predBatch.max(1)[1] == yBatch.nonzero()[:, 1])*isReals).sum()
                totalClean += isReals.sum()
            
            # logging
            accTest = successes/total
            accTestClean = successesClean/totalClean
            print(f'test accuracy: {100*accTest:0.2f} | clean test accuracy: {100*accTestClean:0.2f}%')
    
        # scheduling
        scheduler.step()
    
        # statistics
        statistics.loc[statistics.shape[0]] = [
            e,
            accTrain.to('cpu').item(),
            accTest.to('cpu').item(),
            accTestClean.to('cpu').item(),
            time.perf_counter() - timeStart
        ]
    
    # saving the network
    folderNetwork = folderData+'network/'
    if not os.path.exists(folderNetwork):
        os.makedirs(folderNetwork)
    with open(folderNetwork+'network.pickle', 'wb') as file:
        pickle.dump(
            obj = network.state_dict(),
            file = file,
            protocol = pickle.HIGHEST_PROTOCOL)
    statistics.to_csv(folderNetwork+'statistics.csv', index=False)
    
    # declation (end)
    print('')
    
#%% evaluation
def evaluate(folderBase):
    # imports
    import sys
    import json
    import os
    import pandas as pd
    import pickle
    import torch
    import tqdm
    import extraClasses
    
    # checking files
    folderData = folderBase+'/dsl_data/'
    if not os.path.exists(folderData):
        sys.exit('"dsl_data" directory does not exist!')
    directories = ['metadata', 'batches', 'network']
    for directory in directories:
        if not os.path.exists(folderData+directory):
            sys.exit('"'+directory+'" directory does not exist!')
    
    # loading metadata
    with open(folderData+'metadata/config.json', 'r') as file:
        config = json.load(file)
    with open(folderData+'metadata/batchesEval.json', 'r') as file:
        batches = json.load(file)
    batchNames = list(batches.keys())
    
    # loading batches
    folderBatches = folderData+'batches/'
    XBatches = {}
    yGenderBatches = {}
    yLangFirstBatches = {}
    yLangSecondBatches = {}
    yAgeBatches = {}
    yFluencyBatches = {}
    print('** Loading Evaluation Batches **')
    for batchName in tqdm.tqdm(batchNames):
        with open(folderBatches+batchName+'.pickle', 'rb') as file:
            XBatch, yGenderBatch, yLangFirstBatch, yLangSecondBatch, yAgeBatch, yFluencyBatch = pickle.load(file)
            XBatches[batchName] = XBatch
            yGenderBatches[batchName] = yGenderBatch
            yLangFirstBatches[batchName] = yLangFirstBatch
            yLangSecondBatches[batchName] = yLangSecondBatch
            yAgeBatches[batchName] = yAgeBatch
            yFluencyBatches[batchName] = yFluencyBatch
    
    # loading the network
    folderNetwork = folderData+'network/'
    with open(folderNetwork+'network.pickle', 'rb') as file:
        states = pickle.load(file)
    
    # getting classes
    labelTransformer = extraClasses.LabelTransformer(folderBase)
    
    # evaluation
    # device
    if config['network']['general']['useCuda']:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'
    
    # network
    network = extraClasses.Network(folderBase, device).to(device)
    network.load_state_dict(states)
    
    # making inferences
    network.eval()
    with torch.no_grad():
        result = pd.DataFrame()
        print('** Evaluation **')
        bar = tqdm.tqdm(batchNames)
        for batchName in bar:
        # batches
            XBatch = XBatches[batchName].to(device)
            yGenderBatch = yGenderBatches[batchName].to(device)
            yLangFirstBatch = yLangFirstBatches[batchName].to(device)
            yLangSecondBatch = yLangSecondBatches[batchName].to(device)
            yAgeBatch = yAgeBatches[batchName].to(device)
            yFluencyBatch = yFluencyBatches[batchName].to(device)
            
            # outputs
            _, _, outBatch = network(
                X = XBatch,
                yGender = yGenderBatch,
                yLangFirst = yLangFirstBatch,
                yLangSecond = yLangSecondBatch,
                yAge = yAgeBatch,
                yFluency = yFluencyBatch
                )
            
            # prediction
            predBatch = network.predictor(outBatch)
            
            # summations
            labels = labelTransformer.labelDecode(
                labels = 'intent',
                codes = predBatch)
            labels.index = batches[batchName]['references']
            result = pd.concat([result, labels])
    
    # updating the result
    result.sort_index(inplace=True)
    
    # declation (end)
    print('')
    
    # saving the result
    result.index.name = 'Id'
    result.columns = ['Predicted']
    result.to_csv(folderBase+'resultEvaluation.csv')

def plotResult(folderBase):
    # import
    import pandas as pd
    import matplotlib.pyplot as plt

    # loading the statistics
    statistics = pd.read_csv(folderBase+'dsl_data/network/statistics.csv')

    fig = plt.figure(figsize=(16, 9))
    ax = plt.subplot()
    ax.plot(
        statistics['epoch'],
        statistics['accTrain'],
        label = 'training'
        )
    ax.plot(
        statistics['epoch'],
        statistics['accTestClean'],
        label = 'test'
        )
    ax.set_xlabel('epoch')
    ax.set_ylabel('accuracy')
    ax.set_xlim([statistics['epoch'].iloc[0], statistics['epoch'].iloc[-1]])
    ax.set_ylim([0, 1])
    ax.legend(loc = 'lower right')
    ax.grid()
    plt.show()
    fig.savefig(folderBase+'resultDevelopment.png')