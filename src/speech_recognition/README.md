# EMSConformer

<!-- `conda activate xgb-gpu`
`export PYTHONPATH=/home/liuyi/emsAssist_mobisys22/src/speech_recognition` -->

## 1. Reproduce the Table 3


### 1.1 Google Cloud

We do not provide the access and commands to Google Cloud Speech-to-Text service. Instead, we provide the transcription result text files from Google Cloud for evaluation. Please enter `y` when prompted to rewrite evaluation result file.

<!-- `cd ~/emsAssist_mobisys22/src/speech_recognition` -->

Evaluate Google Cloud on our EMS recordings: `python evaluate_asr_google_cloud.py --dir /home/EMSAssist/data/transcription_text/cloud_translate`

> 		 GC1           0.19617589              0.09608538
> 		 GC2           0.1844242               0.048514143
> 		 GC3           0.17997624              0.05102378
> 		 GC4           0.15486819              0.04636817
> 		 GC5           0.49261895              0.18047477
> 		 GC6           0.20230265              0.098688245
> 		 GC7           0.08300704              0.030546615
> 		 GC8           0.30032083              0.1126812
> 		 This run takes 0:01:28.366960


### 1.2 EMSConformer on Server

Checkout to the conformer directory: `cd examples/conformer`

<!-- `python test.py --output test_outputs/test_for_all_ae.txt --saved /home/EMSAssist/model/speech_models/all_14.h5 --config config_PretrainLibrispeech_TrainEMS_all.yml` -->
(Server) Evaluate provided conformer model on our EMS recordings: `python test.py --output test_outputs/test_for_all_ae.txt --saved ../../../../model/speech_models/Conformer_Pretraining_Fine_Tune_14.h5 --config config_PretrainLibrispeech_TrainEMS_all_updated.yml`

> INFO:tensorflow:greedy_wer: 0.07701108604669571
>
> INFO:tensorflow:greedy_cer: 0.042713455855846405
>
> INFO:tensorflow:beamsearch_wer: 1.0
>
> INFO:tensorflow:beamsearch_cer: 1.0
>
> This run takes 0:07:28.011039

### 1.3 EMSConformer on PH-1

(PH-1) Evaluate provided conformer model on our EMS recordings: `python evaluate_asr.py --result_file ../../../../model/speech_models/all_14_tflite_test_output.tsv`

> INFO:tensorflow:wer: 0.07701108604669571
>
> INFO:tensorflow:cer: 0.04283693805336952
>
> This run takes 0:00:11.633928


To save time for artifact evaluation, here, we have pre-generated the transcribed file `all_14_tflite_test_output.tsv`. If users want to generate the transcription result file with tflite model on their own, use the command below. The numbers may slightly differ from above due to hardware precision differences.

Go to the tflite inference directory: `cd inference`

(PH-1) Evaluate provided conformer tflite model on our EMS recordings: `python run_tflite_model_in_files_easy.py --tflite_model ../../../../../model/speech_models/all_14_model.tflite --data_path ../../../../../data/transcription_text/audio_all/finetune_test_updated.tsv`

> INFO:tensorflow:wer: 0.076571024954319
> 
> INFO:tensorflow:cer: 0.04253384470939636
> 
> This run takes 0:07:19.023407

## 2 Reproduce the Figure 14

<!-- Before we reproduce the Figure 14, we need to update the path of audio files.

Go to the EMSAssist/data directory: `cd ../../data/transcription_text`.

Reconfig the documentation files of audio data path: `python reconfig_data_path.py`.

Go to the EMSAssist/src/speech_recognition directory: `cd ../../src/speech_recognition`.

Make sure you are under the directory: `EMSAssist/src/speech_recognition`.

With the numbers reproduced here with the following commands, you should be able to reproduce the Figure 14. -->

### 2.1 RNN-T Performance with Different Training Strategies

Go the RNNT directory: `cd examples/rnn_transducer`. Make sure you are under directory: `/home/EMSAssist/src/speech_recognition/examples/rnn_transducer` in the container.

* Run RNNT with Pretraining Only strategy (5 mins): `python test.py --saved ../../../../model/speech_models/RNNT_Pretraining_Only_25.h5 --output test_outputs/RNNT_Pretraining_Only.txt --config config_PretrainLibrispeech_DirectInferenceEMS_updated.yml`

> INFO:tensorflow:greedy_wer: 0.8052279353141785
> 
> INFO:tensorflow:greedy_cer: 0.33744192123413086
> 
> INFO:tensorflow:beamsearch_wer: 1.0
> 
> INFO:tensorflow:beamsearch_cer: 1.0
>
> This run takes 0:04:47.780084

* Run RNNT with Fine-tuning Only strategy (5 mins): `python test.py --saved ../../../../model/speech_models/RNNT_Fine_Tune_Only_13.h5 --output test_outputs/RNNT_Fine_Tune_Only.txt --config config_TrainFromScratchEMS_all_updated.yml`

> INFO:tensorflow:greedy_wer: 0.9419996738433838
>
> INFO:tensorflow:greedy_cer: 0.843391478061676
>
> INFO:tensorflow:beamsearch_wer: 1.0
>
> INFO:tensorflow:beamsearch_cer: 1.0
>
> This run takes 0:04:36.110108

* Run RNNT with Prtraining + Fine-tuning strategy (5 mins): `python test.py --saved ../../../../model/speech_models/RNNT_Pretrain_Fine_Tune_02.h5 --output test_outputs/RNNT_Pretrain_Fine_Tune.txt --config config_PretrainLibrispeech_TrainEMS_all_updated.yml`

> INFO:tensorflow:greedy_wer: 0.4103150963783264
>
> INFO:tensorflow:greedy_cer: 0.2703239619731903
>
> INFO:tensorflow:beamsearch_wer: 1.0
>
> INFO:tensorflow:beamsearch_cer: 1.0
>
> This run takes 0:04:36.110108

### 2.2 ContextNet Performance with Different Training Strategies

Go the ContextNet directory: `cd examples/contextnet`. Make sure you are under directory: `EMSAssist/src/speech_recognition/examples/contextnet`

* Run ContextNet with Pretraining Only strategy (3.5 mins): `python test.py --saved ../../../../model/speech_models/ContextNet_Pretrain_Only_1008_86.h5 --output test_outputs/ContextNet_Pretrain_Only.txt --config 1008_config_PretrainLibrispeech_DirectInferenceEMS_updated.yml`

> INFO:tensorflow:greedy_wer: 0.6954761743545532
> 
> INFO:tensorflow:greedy_cer: 0.30140769481658936
> 
> INFO:tensorflow:beamsearch_wer: 1.0
> 
> INFO:tensorflow:beamsearch_cer: 1.0
>
> This run takes 0:03:33.013962

* Run ContextNet with Fine-tuning Only strategy (3.5 mins): `python test.py --saved ../../../../model/speech_models/ContextNet_Fine_Tune_Only_46.h5 --output test_outputs/ContextNet_Fine_Tune_Only.txt --config 1008_config_TrainFromScratchEMS_all_updated.yml`

> INFO:tensorflow:greedy_wer: 0.9418236017227173
>
> INFO:tensorflow:greedy_cer: 0.854740560054779
>
> INFO:tensorflow:beamsearch_wer: 1.0
>
> INFO:tensorflow:beamsearch_cer: 1.0
>
> This run takes 0:03:36.110108

* Run ContextNet with Prtraining + Fine-tuning strategy (3.5 mins): `python test.py --saved ../../../../model/speech_models/ContextNet_Pretrain_Fine_Tune_24.h5 --output test_outputs/ContextNet_Pretrain_Fine_Tune.txt --config 1008_config_PretrainLibrispeech_TrainEMS_all_updated.yml`

> INFO:tensorflow:greedy_wer: 0.5064249038696289
>
> INFO:tensorflow:greedy_cer: 0.4617094397544861
>
> INFO:tensorflow:beamsearch_wer: 1.0
>
> INFO:tensorflow:beamsearch_cer: 1.0
>
> This run takes 0:03:29.110108

### 2.3 Conformer Performance with Different Training Strategies

Go the Conformer directory: `cd examples/conformer`

* Run Conformer with Pretraining Only strategy (6.5 mins): `python test.py --saved ../../../../model/speech_models/Conformer_Pretraining_Only_latest.h5 --output test_outputs/Conformer_Pretraining_Only.txt --config config_PretrainLibrispeech_DirectInferenceEMS_updated.yml`

> INFO:tensorflow:greedy_wer: 0.5386375784873962
> 
> INFO:tensorflow:greedy_cer: 0.2019263207912445
> 
> INFO:tensorflow:beamsearch_wer: 1.0
> 
> INFO:tensorflow:beamsearch_cer: 1.0
>
> This run takes 0:06:25.780084

* Run Conformer with Fine-tuning Only strategy (5 mins): `python test.py --saved ../../../../model/speech_models/Conformer_Fine_Tune_Only_52.h5 --output test_outputs/Conformer_Fine_Tune_Only.txt --config config_TrainFromScratchEMS_all_updated.yml`

> INFO:tensorflow:greedy_wer: 0.9435839056968689
>
> INFO:tensorflow:greedy_cer: 0.7630161046981812
>
> INFO:tensorflow:beamsearch_wer: 1.0
>
> INFO:tensorflow:beamsearch_cer: 1.0
>
> This run takes 0:04:36.110108

* Run Conformer with Prtraining + Fine-tuning Only strategy (5 mins): `python test.py --saved ../../../../model/speech_models/Conformer_Pretraining_Fine_Tune_14.h5 --output test_outputs/Conformer_Pretrain_Fine_Tune.txt --config config_PretrainLibrispeech_TrainEMS_all_updated.yml`

> INFO:tensorflow:greedy_wer: 0.07701108604669571
>
> INFO:tensorflow:greedy_cer: 0.042713455855846405
>
> INFO:tensorflow:beamsearch_wer: 1.0
>
> INFO:tensorflow:beamsearch_cer: 1.0
>
> This run takes 0:06:19.439324
