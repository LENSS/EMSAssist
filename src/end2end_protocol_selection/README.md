# End-to-End Protocol Selection


<!-- `conda activate xgb-gpu` -->

<!-- `cd ~/emsAssist_mobisys22/src/end2end_protocol_selection` -->


### Table 5: Comparing EMSAssist with SOTA (Google Cloud) on the End-to-End (E2E) protocol selection top-1/3/5 accuracy

E2E accuracy for all users (6.5 minutes): `python emsBERT_e2e_table5.py --protocol_model ../../model/emsBERT/FineTune_MobileEnUncase1_Fitted_Desc/0004  --protocol_tflite_model ../../data/ae_text_data/export_tflite/FineTune_MobileEnUncase1_Fitted_Desc_batch1.tflite --cuda_device 0 --transcription_dir ../../data/transcription_text/`

> \##### SOTA E2E Protocol Selection Accuracy #####
>
> Truth: Server [0.14, 0.35, 0.46], PH-1 [0.12, 0.27, 0.32]
>
> GC1: Server [0.14, 0.3, 0.42], PH-1 [0.11, 0.24, 0.3]
>
> GC2: Server [0.13, 0.28, 0.4], PH-1 [0.1, 0.23, 0.29]
>
> GC3: Server [0.13, 0.32, 0.43], PH-1 [0.09, 0.24, 0.29]
>
> GC4: Server [0.14, 0.32, 0.43], PH-1 [0.11, 0.26, 0.31]
>
> GC5: Server [0.12, 0.31, 0.42], PH-1 [0.09, 0.23, 0.3]
>
> GC6: Server [0.13, 0.33, 0.43], PH-1 [0.12, 0.26, 0.31]
>
> GC7: Server [0.1, 0.24, 0.34], PH-1 [0.06, 0.19, 0.26]
>
> GC8: Server [0.14, 0.32, 0.44], PH-1 [0.1, 0.26, 0.32]
>
> \##### EMSMobileBERT E2E Protocol Selection Accuracy #####
>
> Truth: Server [0.73, 0.93, 0.98], PH-1 [0.74, 0.93, 0.98]
>
> E2E: Server [0.71, 0.91, 0.95], PH-1 [0.71, 0.9, 0.95]
>
> This run takes 0:06:31.969176

### Table 6: E2E protocol selection top-1/3/5 accuracy for different users

`python emsBERT_e2e_table6.py --protocol_model ../../model/emsBERT/FineTune_MobileEnUncase1_Fitted_Desc/0004  --protocol_tflite_model ../../data/ae_text_data/export_tflite/FineTune_MobileEnUncase1_Fitted_Desc_batch1.tflite --cuda_device 0 --transcription_dir ../../data/transcription_text/`

> \##### EMSMobileBERT E2E Protocol Selection Accuracy #####
> 
>                  SOTA (GC7)                                      EMSAssist (ours)
> 
>                  Server                  PH-1                    Server                  PH-1
> 
>          Truth   [0.17, 0.33, 0.4]       [0.05, 0.22, 0.27]      [0.72, 0.9, 0.97]       [0.75, 0.9, 0.97]
> 
>          E2E     [0.11, 0.2, 0.28]       [0.06, 0.15, 0.22]      [0.69, 0.88, 0.92]      [0.72, 0.92, 0.92]
> 
>          Truth   [0.15, 0.31, 0.46]      [0.13, 0.24, 0.3]       [0.74, 0.93, 0.98]      [0.74, 0.93, 0.98]
> 
>          E2E     [0.06, 0.18, 0.33]      [0.04, 0.16, 0.21]      [0.72, 0.91, 0.95]      [0.72, 0.95, 0.95]
> 
>          Truth   [0.2, 0.36, 0.52]       [0.14, 0.28, 0.33]      [0.72, 0.89, 0.96]      [0.74, 0.9, 0.96]
> 
>          E2E     [0.1, 0.25, 0.35]       [0.05, 0.21, 0.26]      [0.66, 0.87, 0.96]      [0.69, 0.96, 0.96]
> 
>          Truth   [0.07, 0.35, 0.43]      [0.15, 0.32, 0.37]      [0.78, 0.96, 0.99]      [0.78, 0.96, 0.99]
> 
>          E2E     [0.06, 0.25, 0.33]      [0.05, 0.25, 0.3]       [0.76, 0.94, 0.99]      [0.76, 0.99, 0.99]
> 
>          Truth   [0.18, 0.44, 0.55]      [0.15, 0.25, 0.33]      [0.71, 0.95, 0.97]      [0.72, 0.95, 0.97]
> 
>          E2E     [0.16, 0.37, 0.49]      [0.09, 0.19, 0.3]       [0.7, 0.93, 0.94]       [0.72, 0.95, 0.95]
> 
>          Truth   [0.09, 0.29, 0.38]      [0.11, 0.3, 0.34]       [0.68, 0.93, 0.98]      [0.68, 0.94, 0.98]
> 
>          E2E     [0.09, 0.18, 0.26]      [0.06, 0.17, 0.24]      [0.65, 0.88, 0.96]      [0.66, 0.94, 0.94]
> 
> This run takes 0:15:26.320612
