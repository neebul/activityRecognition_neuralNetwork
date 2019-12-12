# Libraries needed
library(readr)
library(zoo)
library(tidyverse)
library(signal)
library(e1071)
library(doParallel)
library(dplyr)
library(magicfor)



# Load the data
data <- read.csv(file = 'your_file.csv', header = TRUE)



# Assign filter 
f_noise <- signal::butter(4, 0.5,type = "low")
f_g <- signal::butter(4, 0.02, type= "low")
f_peak <- signal::butter(4, 0.06, type = "low")



# Filter data (b = back, t = thigh)
b_x_noise = signal::filtfilt(f_noise, data$x_b)
b_y_noise = signal::filtfilt(f_noise, data$y_b)	
b_z_noise = signal::filtfilt(f_noise, data$z_b)
b_vm_noise = signal::filtfilt(f_noise, data$vec_magnitude_b)

t_x_noise = signal::filtfilt(f_noise, data$x_t)
t_y_noise = signal::filtfilt(f_noise, data$y_t)
t_z_noise = signal::filtfilt(f_noise, data$z_t)
t_vm_noise = signal::filtfilt(f_noise, data$vec_magnitude_t)

filtered_data <- data.frame(b_x_noise = b_x_noise,
                            b_y_noise = b_y_noise,	
                            b_z_noise = b_z_noise,
                            b_vm_noise = b_vm_noise,
                            b_x_g = signal::filtfilt(f_g, filtered_data$b_x_noise),	
                            b_y_g = signal::filtfilt(f_g, filtered_data$b_y_noise),	
                            b_z_g = signal::filtfilt(f_g, filtered_data$b_z_noise),
                            b_vm_g = signal::filtfilt(f_g, filtered_data$b_vm_noise),	
                            b_x_peak = signal::filtfilt(f_peak, filtered_data$b_x_noise),	
                            b_y_peak = signal::filtfilt(f_peak, filtered_data$b_y_noise),	
                            b_z_peak = signal::filtfilt(f_peak, filtered_data$b_z_noise),	
                            b_vm_peak = signal:: filtfilt(f_peak, filtered_data$b_vm_noise),	
                            t_x_noise = t_x_noise,
                            t_y_noise = t_y_noise,	
                            t_z_noise = t_z_noise,	
                            t_vm_noise = t_vm_noise,
                            t_x_g = signal::filtfilt(f_g, filtered_data3$t_x_noise),	
                            t_y_g = signal::filtfilt(f_g, filtered_data3$t_y_noise),	
                            t_z_g = signal::filtfilt(f_g, filtered_data3$t_z_noise),
                            t_vm_g = signal::filtfilt(f_g, filtered_data3$t_vm_noise),	
                            t_x_peak = signal::filtfilt(f_peak, filtered_data3$t_x_noise),	
                            t_y_peak = signal::filtfilt(f_peak, filtered_data3$t_y_noise),	
                            t_z_peak = signal::filtfilt(f_peak, filtered_data3$t_z_noise),	
                            t_vm_peak = signal:: filtfilt(f_peak, filtered_data3$t_vm_noise),
                            stringsAsFactors = FALSE)



# Create functions to deal with peak values
fft_f <- function(v){
  s <- signal::specgram(v, n=length(v), Fs=100)
  S <- abs(s$S)
  f <- S / max(S)
  
  f1 <- f[s$f >= 0.1]
  freq1 <- s$f[s$f >= 0.1]
  d <- ifelse(length(freq1[which.max(f1)])==0, 0, freq1[which.max(f1)])
  p <- ifelse(length(f1)==0, 0, max(f1))
  
  list(dominant=d, power=p, entropy=sum(f * log(f)))
}

find_peaks <- function(v, mindistance, minheight){
  p <- pracma::findpeaks(v, minpeakdistance = mindistance, minpeakheight = minheight)
  ifelse(is.null(p), 0, nrow(p))
}



# Calculate features for each 5 seconds epoch (with 100 Hz accelerometer's data)
magic_for(silent = TRUE)

for (i in (1:nrow(data))) [500] {

  if (step %% 10000 == 0)
  cat('Step:', step, " ")
    
  df <- filtered_data[i:i+499, ]
 
  
    # features 
    b_count = sum(df$b_vm_noise >= 1.2 | df$b_vm_noise <= 0.8)
    t_count = sum(df$t_vm_noise >= 1.2 | df$t_vm_noise <= 0.8)
    
    b_mean_x = mean(df$b_x_g)
    b_mean_y = mean(df$b_y_g)
    b_mean_z = mean(df$b_z_g)
    t_mean_x = mean(df$t_x_g)
    t_mean_y = mean(df$t_y_g)
    t_mean_z = mean(df$t_z_g)
    b_mag_mean = mean(df$b_vm_g)
    t_mag_mean = mean(df$t_vm_g)
    
    b_sd_x = sd(df$b_x_noise)
    b_sd_y = sd(df$b_y_noise)
    b_sd_z = sd(df$b_z_noise)
    t_sd_x = sd(df$t_x_noise)
    t_sd_y = sd(df$t_y_noise)
    t_sd_z = sd(df$t_z_noise)
    b_mag_sd = sd(df$b_vm_noise)
    t_mag_sd = sd(df$t_vm_noise)
    
    b_coef_x = ifelse(b_mean_x > 0, b_sd_x / b_mean_x, 0)
    b_coef_y = ifelse(b_mean_y > 0, b_sd_y / b_mean_y, 0)
    b_coef_z = ifelse(b_mean_z > 0, b_sd_z / b_mean_z, 0)
    b_mag_coef = ifelse(b_mag_mean > 0, b_mag_sd/ b_mag_mean, 0)
    t_coef_x = ifelse(t_mean_x > 0, t_sd_x / t_mean_x, 0)
    t_coef_y = ifelse(t_mean_y > 0, t_sd_y / t_mean_y, 0)
    t_coef_z = ifelse(t_mean_z > 0, t_sd_z / t_mean_z, 0)
    t_mag_coef = ifelse(t_mag_mean > 0, t_mag_sd/ t_mag_mean, 0)
    
    b_quan_25_x = quantile(df$b_x_g, 0.25)
    b_quan_25_y = quantile(df$b_y_g, 0.25)
    b_quan_25_z = quantile(df$b_z_g, 0.25)
    t_quan_25_x = quantile(df$t_x_g, 0.25)
    t_quan_25_y = quantile(df$t_y_g, 0.25)
    t_quan_25_z = quantile(df$t_z_g, 0.25)
    b_mag_quan_25 = quantile(df$b_vm_g, 0.25)
    t_mag_quan_25 = quantile(df$t_vm_g, 0.25)
    
    b_quan_50_x = quantile(df$b_x_g, 0.50)
    b_quan_50_y = quantile(df$b_y_g, 0.50)
    b_quan_50_z = quantile(df$b_z_g, 0.50)
    t_quan_50_x = quantile(df$t_x_g, 0.50)
    t_quan_50_y = quantile(df$t_y_g, 0.50)
    t_quan_50_z = quantile(df$t_z_g, 0.50)
    b_mag_quan_50 = quantile(df$b_vm_g, 0.50)
    t_mag_quan_50 = quantile(df$t_vm_g, 0.50)
    
    b_quan_75_x = quantile(df$b_x_g, 0.75)
    b_quan_75_y = quantile(df$b_y_g, 0.75)
    b_quan_75_z = quantile(df$b_z_g, 0.75)
    t_quan_75_x = quantile(df$t_x_g, 0.75)
    t_quan_75_y = quantile(df$t_y_g, 0.75)
    t_quan_75_z = quantile(df$t_z_g, 0.75)
    b_mag_quan_75 = quantile(df$b_vm_g, 0.75)
    t_mag_quan_75 = quantile(df$t_vm_g, 0.75)
    
    b_min_x = min(df$b_x_g)
    b_min_y = min(df$b_y_g)
    b_min_z = min(df$b_z_g)
    t_min_x = min(df$t_x_g)
    t_min_y = min(df$t_y_g)
    t_min_z = min(df$t_z_g)
    b_mag_min = min(df$b_vm_g)
    t_mag_min = min(df$t_vm_g)
    
    b_max_x = max(df$b_x_g)
    b_max_y = max(df$b_y_g)
    b_max_z = max(df$b_z_g)
    t_max_x = max(df$t_x_g)
    t_max_y = max(df$t_y_g)
    t_max_z = max(df$t_z_g)
    b_mag_max = max(df$b_vm_g)
    t_mag_max = max(df$t_vm_g)
    
    corr_xTyT = ifelse(t_sd_x>0 & t_sd_y>0, cor(df$t_x_noise, df$t_y_noise),0)
    corr_xTzT = ifelse(t_sd_x>0 & t_sd_z>0, cor(df$t_x_noise, df$t_z_noise),0)
    corr_yTzT = ifelse(t_sd_y>0 & t_sd_z>0, cor(df$t_y_noise, df$t_z_noise),0)
    corr_xByB = ifelse(b_sd_x>0 & b_sd_y>0, cor(df$b_x_noise, df$b_y_noise),0)
    corr_xBzB = ifelse(b_sd_x>0 & b_sd_z>0, cor(df$b_x_noise, df$b_z_noise),0)
    corr_yBzB = ifelse(b_sd_y>0 & b_sd_z>0, cor(df$b_z_noise, df$b_y_noise),0)
    corr_xTxB = ifelse(t_sd_x>0 & b_sd_x>0, cor(df$t_x_noise, df$b_x_noise),0)
    corr_xTyB = ifelse(t_sd_x>0 & b_sd_y>0, cor(df$t_x_noise, df$b_y_noise),0)
    corr_xTzB = ifelse(t_sd_x>0 & b_sd_z>0, cor(df$t_x_noise, df$b_z_noise),0)
    corr_yTxB = ifelse(t_sd_y>0 & b_sd_x>0, cor(df$t_y_noise, df$b_x_noise),0)
    corr_yTyB = ifelse(t_sd_y>0 & b_sd_y>0, cor(df$t_y_noise, df$b_y_noise),0)
    corr_yTzB = ifelse(t_sd_y>0 & b_sd_z>0, cor(df$t_y_noise, df$b_z_noise),0)
    corr_zTxB = ifelse(t_sd_z>0 & b_sd_x>0, cor(df$t_z_noise, df$b_x_noise),0)
    corr_zTyB = ifelse(t_sd_z>0 & b_sd_y>0, cor(df$t_z_noise, df$b_y_noise),0)
    corr_zTzB = ifelse(t_sd_z>0 & b_sd_z>0, cor(df$t_z_noise, df$b_z_noise),0)
    corr_mTmB = ifelse(t_mag_sd>0 & b_mag_sd>0,cor(df$b_vm_noise, df$t_vm_noise),0)
    
    mean_xBxT = sum(t_mean_x, b_mean_x)
    mean_yByT = sum(t_mean_y, b_mean_y)
    mean_zBzT = sum(t_mean_z, b_mean_z)
    mean_xByT = sum(t_mean_y, b_mean_x)
    mean_xBzT = sum(t_mean_z, b_mean_x)
    mean_yBzT = sum(t_mean_z, b_mean_y)
    mean_yBxT = sum(t_mean_x, b_mean_y)
    mean_zBxT = sum(t_mean_x, b_mean_z)
    mean_zByT = sum(t_mean_y, b_mean_z)
    
    t_roll_mean = mean(atan2(df$t_y_g, df$t_x_g))
    b_roll_mean = mean(atan2(df$b_y_g, df$b_x_g))
    t_pitch_mean = mean(atan2(df$t_x_g, df$t_z_g))
    b_pitch_mean = mean(atan2(df$b_x_g, df$b_z_g))
    t_yaw_mean = mean(atan2(df$t_y_g, df$t_z_g))
    b_yaw_mean = mean(atan2(df$b_y_g, df$b_z_g))
    
    t_roll_sd = sd(atan2(df$t_y_g, df$t_x_g))
    b_roll_sd = sd(atan2(df$b_y_g, df$b_x_g))
    t_pitch_sd = sd(atan2(df$t_x_g, df$t_z_g))
    b_pitch_sd = sd(atan2(df$b_x_g, df$b_z_g))
    t_yaw_sd = sd(atan2(df$t_y_g, df$t_z_g))
    b_yaw_sd = sd(atan2(df$b_y_g, df$b_z_g))
    
    t_mean_g_x = mean(df$t_x_g)
    t_mean_g_y = mean(df$t_y_g)
    t_mean_g_z = mean(df$t_z_g)
    b_mean_g_x = mean(df$b_x_g)
    b_mean_g_y = mean(df$b_y_g)
    b_mean_g_z = mean(df$b_z_g)
    t_mag_g_mean = mean(df$t_vm_g)
    b_mag_g_mean = mean(df$b_vm_g)
    
    t_fft_freq_x = fft_f(df$t_x_peak)$dominant
    t_fft_freq_y = fft_f(df$t_y_peak)$dominant
    t_fft_freq_z = fft_f(df$t_z_peak)$dominant
    t_fft_pow_x = fft_f(df$t_x_peak)$power
    t_fft_pow_y = fft_f(df$t_y_peak)$power
    t_fft_pow_z = fft_f(df$t_z_peak)$power
    t_fft_mag_freq = fft_f(df$t_vm_peak)$dominant
    t_fft_mag_pow = fft_f(df$t_vm_peak)$power
  
    b_fft_freq_x = fft_f(df$b_x_peak)$dominant
    b_fft_freq_y = fft_f(df$b_y_peak)$dominant
    b_fft_freq_z = fft_f(df$b_z_peak)$dominant
    b_fft_pow_x = fft_f(df$b_x_peak)$power
    b_fft_pow_y = fft_f(df$b_y_peak)$power
    b_fft_pow_z = fft_f(df$b_z_peak)$power
    b_fft_mag_freq = fft_f(df$b_vm_peak)$dominant
    b_fft_mag_pow = fft_f(df$b_vm_peak)$power
    
    t_peaks = min (find_peaks(df$t_x_peak, 20, 0.02), find_peaks(df$t_y_peak, 20, 0.02), find_peaks(df$t_z_peak, 20, 0.02))
    b_peaks = min (find_peaks(df$b_x_peak, 20, 0.02), find_peaks(df$b_y_peak, 20, 0.02), find_peaks(df$b_z_peak, 20, 0.02))
    
    t_kurtosis_x = e1071::kurtosis(df$t_x_g)
    t_kurtosis_y = e1071::kurtosis(df$t_y_g)
    t_kurtosis_z = e1071::kurtosis(df$t_z_g)
    b_kurtosis_x = e1071::kurtosis(df$b_x_g)
    b_kurtosis_y = e1071::kurtosis(df$b_y_g)
    b_kurtosis_z = e1071::kurtosis(df$b_z_g)
    
    t_skew_x = e1071::skewness(df$t_x_g)
    t_skew_y = e1071::skewness(df$t_y_g)
    t_skew_z = e1071::skewness(df$t_z_g)
    b_skew_x = e1071::skewness(df$b_x_g)
    b_skew_y = e1071::skewness(df$b_y_g)
    b_skew_z = e1071::skewness(df$b_z_g)
    
    put(b_count, t_count, b_mean_x, b_mean_y, b_mean_z, t_mean_x, t_mean_y, t_mean_z, b_mag_mean, t_mag_mean, b_sd_x, 
        b_sd_y, b_sd_z, t_sd_x, t_sd_y, t_sd_z, b_mag_sd, t_mag_sd, b_coef_x, b_coef_y, b_coef_z, b_mag_coef, t_coef_x, 
        t_coef_y, t_coef_z, t_mag_coef, b_quan_25_x, b_quan_25_y, b_quan_25_z, t_quan_25_x, t_quan_25_y, t_quan_25_z, 
        b_mag_quan_25, t_mag_quan_25, b_quan_50_x, b_quan_50_y, b_quan_50_z, t_quan_50_x, t_quan_50_y, t_quan_50_z, 
        b_mag_quan_50, t_mag_quan_50, b_quan_75_x, b_quan_75_y, b_quan_75_z, t_quan_75_x, t_quan_75_y, t_quan_75_z, 
        b_mag_quan_75, t_mag_quan_75, b_min_x, b_min_y, b_min_z, t_min_x, t_min_y, t_min_z, b_mag_min, t_mag_min, 
        b_max_x, b_max_y, b_max_z, t_max_x, t_max_y, t_max_z, b_mag_max, t_mag_max, corr_xTyT, corr_xTzT, corr_yTzT, 
        corr_xByB, corr_xBzB, corr_yBzB, corr_xTxB, corr_xTyB, corr_xTzB, corr_yTxB, corr_yTyB, corr_yTzB, corr_zTxB, 
        corr_zTyB, corr_zTzB, corr_mTmB, mean_xBxT, mean_yByT, mean_zBzT, mean_xByT, mean_xBzT, mean_yBzT, mean_yBxT, 
        mean_zBxT, mean_zByT, t_roll_mean, b_roll_mean, t_pitch_mean, b_pitch_mean, t_yaw_mean, b_yaw_mean, t_roll_sd, 
        b_roll_sd, t_pitch_sd, b_pitch_sd, t_yaw_sd, b_yaw_sd, t_mean_g_x, t_mean_g_y, t_mean_g_z, b_mean_g_x, b_mean_g_y, 
        b_mean_g_z, t_mag_g_mean, b_mag_g_mean, t_fft_freq_x, t_fft_freq_y, t_fft_freq_z, t_fft_pow_x, t_fft_pow_y, t_fft_pow_z, 
        t_fft_mag_freq, t_fft_mag_pow, b_fft_freq_x, b_fft_freq_y, b_fft_freq_z, b_fft_pow_x, b_fft_pow_y, b_fft_pow_z, 
        b_fft_mag_freq, b_fft_mag_pow, t_peaks, b_peaks, t_kurtosis_x, t_kurtosis_y, t_kurtosis_z, b_kurtosis_x, b_kurtosis_y, 
        b_kurtosis_z, t_skew_x, t_skew_y, t_skew_z, b_skew_x, b_skew_y, b_skew_z) 
    
}


temporary_features <- magic_result_as_dataframe()
