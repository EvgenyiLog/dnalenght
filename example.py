import pandas as pd
import numpy as np
from readerfrf import parse_frf_file
import matplotlib.pyplot as plt
from subtract_reference_from_columns import subtract_reference_from_columns
from msbackadj import msbackadj
from pybaselines import Baseline
from categorize_frf_files import categorize_frf_files
from score_peaks import score_peaks_genlib
from correct_baseline import correct_baseline
from plot_peak_analysis import plot_peak_analysis
# import pyyawt
from reveal_paths import reveal_paths, extract_paths_from_categorize


def main():
    matrix_df, channels_df, metadata = parse_frf_file(
        r"C:\Users\Admin\Documents\GitHub\dnalenght\files\Anton_lib_test_2_17_56_29\0.1-5-0.2_F9.frf")
    # print(metadata.keys())
    matrix_df, channels_df, metadata = parse_frf_file(
        r"C:\Users\Admin\Downloads\Telegram Desktop\anton_lib_test\Anton_lib_test_2_17_56_29\ladder_A6.frf")
    # print(metadata.keys())
    
    
    print(metadata.get('Size'))
    df=score_peaks_genlib(channels_df.loc[:, 'dR110'])
    plot_peak_analysis(channels_df.loc[:, 'dR110'],df['Index'],df['Mark'],df[df['Selected'] == '✓'].index.tolist())
    keyword_files, other_files = categorize_frf_files(
        input_path=r"C:\Users\Admin\Downloads\Telegram Desktop\anton_lib_test")
    keyword_files, other_files = categorize_frf_files(
        input_path=r"C:\Users\Admin\Downloads\Telegram Desktop\anton_lib_test\Anton_lib_test_3_19_20_20")
    keyword_paths, other_paths = extract_paths_from_categorize(
        keyword_files, other_files)
    path_keyword_files = reveal_paths(keyword_paths)
    path_other_files = reveal_paths(other_paths)
    
    print(metadata.get('Title'))
    # print(channels_df.columns)
    plt.figure('Raw') 
    channels_df.loc[:, 'dR110'].plot(color='b')
    plt.grid(True)

    df = subtract_reference_from_columns(channels_df, 50)
    plt.figure('Вычитание первых 50') 
    df.loc[:, 'dR110'].plot(color='m')
    plt.grid(True)

    time = np.arange(len(channels_df))  # номер отсчёта

    # исходный сигнал
    signal = df['dR110']

    # применяем msbackadj
    signal_corrected = msbackadj(time, signal.values)

    # добавляем в тот же DataFrame
    df['dR110'] = signal_corrected
    plt.figure('msbackadj') 
    df.loc[:, 'dR110'].plot(color='g')
    plt.grid(True)

    corectbaseline,baseline=correct_baseline(df['dR110'].values, method='modpoly',poly_order=1)
    corectbaseline,baseline=correct_baseline(corectbaseline,   method='psalsa',lam=1e5,k=0.05 )
    
    corectbaseline,baseline=correct_baseline(corectbaseline,method='beads',freq_cutoff=0.002,asymmetry=3,lam_0=3,lam_1=0.05,lam_2=0.2)
    corectbaseline,baseline=correct_baseline(corectbaseline, method='iasls',lam=1e6)
    df['dR110'] =  corectbaseline
    plt.figure('corect baseline') 
    df.loc[:, 'dR110'].plot(color='k')
    plt.grid(True)







   
  
    
    
    print(f"Длина исходного сигнала: {len(signal)}")
    # [signal_corrected,CXD,LXD] = pyyawt.wden(signal,'sqtwolog','s','sln',1,'sym2')
    # print(f"Длина результата: {len(signal_corrected)}")
    # df['dR110_corr'] = signal_corrected
    # plt.figure() 
    # df.loc[:,'dR110'].plot(color='k')
    # plt.grid(True)
                                   



    plt.show()


if __name__ == "__main__":
    main()
