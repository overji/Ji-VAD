from modelscope import snapshot_download
snapshot_download('iic/CosyVoice2-0.5B', local_dir='/obs/xuke/cosyvoicepretrained_models/CosyVoice2-0.5B')
snapshot_download('iic/CosyVoice-300M', local_dir='/obs/xuke/cosyvoicepretrained_models/CosyVoice-300M')
snapshot_download('iic/CosyVoice-300M-SFT', local_dir='/obs/xuke/cosyvoicepretrained_models/CosyVoice-300M-SFT')
snapshot_download('iic/CosyVoice-300M-Instruct', local_dir='/obs/xuke/cosyvoicepretrained_models/CosyVoice-300M-Instruct')
snapshot_download('iic/CosyVoice-ttsfrd', local_dir='/obs/xuke/cosyvoicepretrained_models/CosyVoice-ttsfrd')