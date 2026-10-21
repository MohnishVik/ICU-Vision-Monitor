# models/finetuned/

Place your subject-video fine-tuned weight files here:

| File                              | Module | Notes                                    |
|-----------------------------------|--------|------------------------------------------|
| rr_tcn_finetuned.pth              | RR     | LOCKED — fine-tuned on own D435 clips    |
| hr_efficientphys_finetuned.pth    | HR     | LOCKED — fine-tuned on own D435 clips    |
| fall_model_finetuned.pt           | Fall   | LOCKED — your existing fine-tuned .pt    |
| pain_deepmlp_synpain.pth          | Pain   | 77.1% acc — best current pain weight     |
| pain_temporal_tcn.pth             | Pain   | PENDING — add when UNBC/BioVid ready     |
