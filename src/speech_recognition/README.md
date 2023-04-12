# EMSConformer

`conda activate xgb-gpu`
`export PYTHONPATH=/home/liuyi/emsAssist_mobisys22/src/speech_recognition`


### Google Cloud Speech-to-Text

We do not provide the access and commands to Google Cloud Speech-to-Text service. Instead, we provide the transcription result text files from Google Cloud for evaluation. 

`cd ~/emsAssist_mobisys22/src/speech_recognition`

`python evaluate_asr_google_cloud.py --dir /home/liuyi/emsAssist_mobisys22/data/transcription_text/cloud_translate`

> 		 GC1           0.19617589              0.09608538
> 		 GC2           0.1844242               0.048514143
> 		 GC3           0.17997624              0.05102378
> 		 GC4           0.15486819              0.04636817
> 		 GC5           0.49261895              0.18047477
> 		 GC6           0.20230265              0.098688245
> 		 GC7           0.08300704              0.030546615
> 		 GC8           0.30032083              0.1126812
> 		 This run takes 0:01:28.366960


### EMSConformer on Server

`cd examples/conformer`

`python test.py --output test_outputs/test_for_all_ae.txt --saved ~/emsAssist_mobisys22/model/speech_models/all_14.h5 --config config_PretrainLibrispeech_TrainEMS_all.yml`

> INFO:tensorflow:greedy_wer: 0.07701108604669571
>
> INFO:tensorflow:greedy_cer: 0.042713455855846405
>
> INFO:tensorflow:beamsearch_wer: 1.0
>
> INFO:tensorflow:beamsearch_cer: 1.0
>
> This run takes 0:07:28.011039

### EMSConformer on PH-1

`python evaluate_asr.py --result_file /home/liuyi/emsAssist_mobisys22/model/speech_models/all_14_tflite_test_output.tsv`

> INFO:tensorflow:wer: 0.07701108604669571
>
> INFO:tensorflow:cer: 0.04283693805336952
>
> This run takes 0:00:11.633928


To save time for artifact evaluation, here, we have pre-generated the transcripted file `all_14_tflite_test_output.tsv`. If users want to generate the transcription result file with tflite model on their own, use the command below:

`cd inference`

`python run_tflite_model_in_files_easy.py --tflite_model /home/liuyi/emsAssist_mobisys22/model/speech_models/all_14_model.tflite`

> INFO:tensorflow:wer: 0.076571024954319
> 
> INFO:tensorflow:cer: 0.04253384470939636
> 
> This run takes 0:07:19.023407
