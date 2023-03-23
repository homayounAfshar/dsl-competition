#%% imports
import json
import librosa
import audiomentations
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchaudio

#%% label transformer
class LabelTransformer():
    def __init__(self, folderBase):
        # loading metadata
        with open(folderBase+'/dsl_data/metadata/labels.json', 'r') as file:
            labels = json.load(file)
        
        # loading up the label transformer
        self.labels = labels
        
        # encoders and decoders
        for label, specs in labels.items():
            size = specs['size']
            values = specs['values']
            encoder = pd.Series(
                index = values,
                data = [i for i in range(size)]
                )
            decoder = pd.Series(
                index = [i for i in range(size)],
                data = values
                )
            setattr(self, label+'Encoder', encoder)
            setattr(self, label+'Decoder', decoder)

    def labelEncode(self, dataFrame):
        output = []
        for label in dataFrame.columns:
            encoder = getattr(self, label+'Encoder')
            codes = torch.scatter(
                torch.zeros(dataFrame.shape[0], len(self.labels[label]['values'])),
                1,
                torch.tensor(encoder.loc[dataFrame[label]]).reshape(dataFrame.shape[0], 1),
                1
                )
            output.append(codes)
        return tuple(output)
    
    def labelDecode(self, labels, codes):
        if type(labels) != list and type(labels) != tuple:
            labels = [labels]
            codes = [codes]
        data = {}
        for i in range(len(labels)):
            decoder = getattr(self, labels[i]+'Decoder')
            data[labels[i]] = decoder.loc[codes[i].to(torch.float).max(1)[1].tolist()].tolist()
        dataFrame = pd.DataFrame(data)
        return dataFrame

#%% reader
class Reader():
    def __init__(self, folderBase):
        # loading metadata
        with open(folderBase+'/dsl_data/metadata/config.json', 'r') as file:
            config = json.load(file)
            
        # loading up the reader
        self.config = config
        
    def __call__(self, path):
        waveform = librosa.load(
            path = path,
            sr = self.config['preprocessing']['reader']['sampleRate'])[0]
        mask = (np.abs(waveform) > self.config['preprocessing']['reader']['trimThreshold']*np.abs(waveform).max()).nonzero()[0]
        return waveform[mask[0]:mask[-1]+1]

#%% augmenter
class Augmenter():
    def __init__(self, folderBase):
        # loading metadata
        with open(folderBase+'/dsl_data/metadata/config.json', 'r') as file:
            config = json.load(file)
            
        # loading up the augmenter
        self.config = config
        self.augment = audiomentations.Compose([
            audiomentations.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            audiomentations.TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
            audiomentations.PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
            audiomentations.Shift(min_fraction=0, max_fraction=0.2, p=0.5),
        ])
        
    def __call__(self, waveform, nTries):
        waveforms = []
        lengths = []
        for i in range(nTries):
            waveformAugmented = self.augment(waveform, self.config['preprocessing']['reader']['sampleRate'])
            waveforms.append(waveformAugmented)
            lengths.append(waveformAugmented.shape[0])
        return waveforms, lengths

#%% aligner
class Aligner():
    def __init__(self, folderBase):
        # loading metadata
        with open(folderBase+'/dsl_data/metadata/config.json', 'r') as file:
            config = json.load(file)
        
        # loading up the aligner
        self.config = config
        
    def __call__(self, waveform, length):
        if waveform.shape[0] < length:
            waveformAligned = np.zeros(length)
            waveformAligned[:waveform.shape[0]] = waveform
        else:
            waveformAligned = waveform[:length]
        return waveformAligned

#%% feature extractor
class FeatureExtractor():
    def __init__(self, folderBase):
        # loading metadata
        with open(folderBase+'/dsl_data/metadata/config.json', 'r') as file:
            config = json.load(file)
            
        # loading up the feature extractor
        self.config = config
        
    def __call__(self, waveforms):
        if waveforms.ndim == 1:
            waveforms = waveforms.reshape(1, waveforms.shape[0])
        getMfcc = torchaudio.transforms.MFCC(
            sample_rate = self.config['preprocessing']['reader']['sampleRate'],
            n_mfcc = self.config['preprocessing']['featureExtractor']['nMfcc'],
            melkwargs = {
                'n_fft': self.config['preprocessing']['featureExtractor']['nFft'],
                'hop_length': self.config['preprocessing']['featureExtractor']['nHop'],
                }
            )
        features = getMfcc(waveforms)
        if self.config['preprocessing']['featureExtractor']['doNormalize']:
            features = (features - features.mean(axis=2, keepdims=True))/features.std(axis=2, keepdims=True)
        return features.permute([0, 2, 1])
        
#%% network
class Network(nn.Module):
    def __init__(self, folderBase, device):
        # initialization
        super(Network, self).__init__()
        
        # loading metadata
        with open(folderBase+'/dsl_data/metadata/config.json', 'r') as file:
            config = json.load(file)
        with open(folderBase+'/dsl_data/metadata/labels.json', 'r') as file:
            labels = json.load(file)
        sizeInputs = {
            'stack': config['preprocessing']['featureExtractor']['nMfcc'],
            'nGenders': labels['gender']['size'],
            'nLangsFirst': labels['langFirst']['size'],
            'nLangsSecond': labels['langSecond']['size'],
            'nAges': labels['age']['size'],
            'nFulencies': labels['fluency']['size'],
            'nActions': labels['action']['size'],
            'nObjects': labels['object']['size'],
            'nIntents': labels['intent']['size']
            }
        
        # loading up the network
        self.config = config
        self.device = device
        self.sizeInputs = sizeInputs
        
        # stack
        self.stack = nn.LSTM(
            input_size = sizeInputs['stack'],
            hidden_size = config['network']['stack']['nHidden'],
            num_layers = config['network']['stack']['nLayers'],
            bidirectional = config['network']['stack']['isBidirectional'],
            batch_first = True
            )
        
        # representatives
        self.representativeAction = nn.LSTM(
            input_size = (config['network']['stack']['isBidirectional'] + 1)*config['network']['stack']['nHidden'],
            hidden_size = config['network']['representative']['nHidden'],
            num_layers = config['network']['representative']['nLayers'],
            bidirectional = config['network']['representative']['isBidirectional'],
            batch_first = True
            )
        self.representativeObject = nn.LSTM(
            input_size = (config['network']['stack']['isBidirectional'] + 1)*config['network']['stack']['nHidden'],
            hidden_size = config['network']['representative']['nHidden'],
            num_layers = config['network']['representative']['nLayers'],
            bidirectional = config['network']['representative']['isBidirectional'],
            batch_first = True
            )
        
        # dropout layers
        self.dropout0Action = nn.Dropout(
            p = config['network']['general']['probDropout'],
            inplace = False
            )
        self.dropout0Object = nn.Dropout(
            p = config['network']['general']['probDropout'],
            inplace = False
            )
        
        # normalizers
        self.normalizerAction = nn.BatchNorm1d(
            num_features = (config['network']['representative']['isBidirectional'] + 1)*config['network']['representative']['nHidden']
            )
        self.normalizerObject = nn.BatchNorm1d(
            num_features = (config['network']['representative']['isBidirectional'] + 1)*config['network']['representative']['nHidden']
            )
        
        # aggregators
        # self.aggregatorAction = lambda x: x.mean(dim=1)
        # self.aggregatorObject = lambda x: x.mean(dim=1)
        self.aggregatorAction = lambda x: x.max(dim=1)[0]
        self.aggregatorObject = lambda x: x.max(dim=1)[0]
        
        # dropout layers
        self.dropout1Action = nn.Dropout(
            p = config['network']['general']['probDropout'],
            inplace = False
            )
        self.dropout1Object = nn.Dropout(
            p = config['network']['general']['probDropout'],
            inplace = False
            )
        
        # concatenation
        sizeInputsTransformer = (config['network']['representative']['isBidirectional'] + 1)*config['network']['representative']['nHidden']
        sizeInputsTransformer += sizeInputs['nGenders']
        sizeInputsTransformer += sizeInputs['nLangsFirst']
        sizeInputsTransformer += sizeInputs['nLangsSecond']
        sizeInputsTransformer += sizeInputs['nAges']
        sizeInputsTransformer += sizeInputs['nFulencies']
        
        # transformers
        self.transformerAction = nn.Linear(
            in_features = sizeInputsTransformer,
            out_features = sizeInputs['nActions']
            )
        self.transformerObject = nn.Linear(
            in_features = sizeInputsTransformer + config['network']['sampler']['isActive']*sizeInputs['nActions'],
            out_features = sizeInputs['nObjects']
            )

        # predictor
        self.predictor = nn.Softmax(-1)

        # sampler
        self.sampler = lambda x: config['network']['sampler']['prob']*(1.5-torch.sigmoid(torch.tensor(x)))

        # combiner
        self.combiner = nn.Linear(
            in_features = sizeInputs['nActions'] + sizeInputs['nObjects'],
            out_features = sizeInputs['nIntents']
            )

    def forward(self, X, yGender, yLangFirst, yLangSecond, yAge, yFluency, yAction=None, epoch=None):
        sizeBatch = X.shape[0]
        
        # stack
        out, _ = self.stack(X, self.getHidden('stack', sizeBatch))
        
        # representatives
        outAction, _ = self.representativeAction(out, self.getHidden('representative', sizeBatch))
        outObject, _ = self.representativeObject(out, self.getHidden('representative', sizeBatch))
        
        # dropout layers
        outAction = self.dropout0Action(outAction)
        outObject = self.dropout0Object(outObject)
        
        # normalizers
        outAction = self.normalizerAction(outAction.permute([0, 2, 1])).permute([0, 2, 1])
        outObject = self.normalizerObject(outObject.permute([0, 2, 1])).permute([0, 2, 1])
        
        # aggregators
        outAction = self.aggregatorAction(outAction)
        outObject = self.aggregatorObject(outObject)
        
        # concatenation (action)
        outAction = torch.hstack([outAction, yGender, yLangFirst, yLangSecond, yAge, yFluency])
        
        # dropout layer (action)
        outAction = self.dropout1Action(outAction)
        
        # transformer (action)
        outAction = self.transformerAction(outAction)

        # concatenation (object)
        if self.config['network']['sampler']['isActive']:
            predAction = self.predictor(outAction)
            yEstimatedAction = torch.scatter(
                torch.zeros(sizeBatch, self.sizeInputs['nActions']).to(self.device),
                1,
                predAction.max(1, True)[1],
                1
                )
            if yAction != None:
                probTruth = self.sampler(epoch)
                decider = (torch.rand(sizeBatch, 1) <= probTruth).repeat(1, yAction.shape[1])
                decider = decider.to(self.device)
                yDecidedAction = decider*yAction + ~decider*yEstimatedAction
            else:
                yDecidedAction = yEstimatedAction
            outObject = torch.hstack([outObject, yGender, yLangFirst, yLangSecond, yAge, yFluency, yDecidedAction])
        else:
            outObject = torch.hstack([outObject, yGender, yLangFirst, yLangSecond, yAge, yFluency])
        
        # dropout layer (object)
        outObject = self.dropout1Object(outObject)
        
        # transformer (object)
        outObject = self.transformerObject(outObject)

        # combiner
        out = self.combiner(torch.hstack([outAction, outObject]))

        # output
        return outAction, outObject, out
        
    def getHidden(self, layer, sizeBatch):
        out = (
            torch.zeros(
                (self.config['network'][layer]['isBidirectional'] + 1)*self.config['network'][layer]['nLayers'],
                sizeBatch,
                self.config['network'][layer]['nHidden']
                ).to(self.device),
            torch.zeros(
                (self.config['network'][layer]['isBidirectional'] + 1)*self.config['network'][layer]['nLayers'],
                sizeBatch,
                self.config['network'][layer]['nHidden']
                ).to(self.device)
            )
        return out