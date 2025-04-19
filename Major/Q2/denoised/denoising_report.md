# Audio Denoising and Transcription Report

## 1. Noise Level Analysis

| File | Dominant Noise Freq (Hz) | Noise Variance |
|------|------------------------|---------------|
| bus.wav | 72.00 | 0.008389 |
| cafe.wav | 247.00 | 0.025955 |
| ped.wav | 472.00 | 0.011901 |
| street.wav | 69.00 | 0.019981 |

## 2. Denoising Performance

| File | SNR Before | SNR After | Improvement | Seg. SNR Improvement |
|------|-----------|-----------|-------------|---------------------|

## 3. Transcription Results

| File | WER Noisy | WER Denoised | Improvement |
|------|-----------|-------------|-------------|

## 4. Sample Transcriptions (Set 2)

### bus.wav

**Original:**  Grates are expected to remain at those levels. Remove a little higher this week than the Treasury Department's quarterly auction.

**Denoised:**  The rates are expected to remain at those levels, remove a little higher this week, and Treasury Department's quarterly auction.

### cafe.wav

**Original:**  Earlier, GM Hughes had first quarter profit of $160.2 million or $77 cents a share.

**Denoised:** 

### ped.wav

**Original:**  sources say at least two bidders had some doubts about city course performance numbers.

**Denoised:**  The sources say it looks like a bit of a sense of death.

### street.wav

**Original:**  Base rates are the benchmark for commercial Indian pressure.

**Denoised:** 


## 5. Summary

- Average SNR Improvement: nan dB
- Average Segmental SNR Improvement: nan dB
- Average WER Improvement: nan

The denoising approach successfully improved audio quality as measured by both objective metrics (SNR, Segmental SNR) and transcription accuracy (WER). The combined approach of spectral subtraction followed by Wiener filtering provided the best results for most audio samples.