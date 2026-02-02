import pandas as pd
import numpy as np
from readerfrf import parse_frf_file
import matplotlib.pyplot as plt
from subtract_reference_from_columns import subtract_reference_from_columns
from msbackadj import msbackadj
from pybaselines import Baseline
from categorize_frf_files import categorize_frf_files
# import pyyawt
from reveal_paths import reveal_paths,extract_paths_from_categorize
def main():
    matrix_df, channels_df, metadata=parse_frf_file( r"C:\Users\Admin\Documents\GitHub\dnalenght\files\Anton_lib_test_2_17_56_29\0.1-5-0.2_F9.frf")
    matrix_df, channels_df, metadata=parse_frf_file( r"C:\Users\Admin\Downloads\Telegram Desktop\anton_lib_test\Anton_lib_test_2_17_56_29\ladder_A6.frf")
    keyword_files, other_files=categorize_frf_files(input_path=r"C:\Users\Admin\Downloads\Telegram Desktop\anton_lib_test")
    keyword_files, other_files=categorize_frf_files(input_path=r"C:\Users\Admin\Downloads\Telegram Desktop\anton_lib_test\Anton_lib_test_3_19_20_20")
    keyword_paths, other_paths = extract_paths_from_categorize(keyword_files, other_files)
    path_keyword_files = reveal_paths(keyword_paths)
    path_other_files = reveal_paths(other_paths)
    # print(metadata.keys())
    print(metadata.get('Title'))
    print(channels_df.columns)
    plt.figure('Raw') 
    channels_df.loc[:,'dR110'].plot(color='b')
    plt.grid(True)


    df=subtract_reference_from_columns(channels_df,50)
    plt.figure('Вычитание первых 50') 
    df.loc[:,'dR110'].plot(color='m')
    plt.grid(True)

    time = np.arange(len(channels_df))  # номер отсчёта

    # исходный сигнал
    signal = df['dR110']

    # применяем msbackadj
    signal_corrected = msbackadj(time, signal.values)

    # добавляем в тот же DataFrame
    df['dR110'] = signal_corrected
    plt.figure('msbackadj') 
    df.loc[:,'dR110'].plot(color='g')
    plt.grid(True)

    x = np.arange(len(df))  # или реальная ось (время/длина волны), если есть
    y = df['dR110'].values

    # Инициализация корректора базовой линии
    baseline_fitter = Baseline(x_data=x)

    # Метод 1: iARPLS (рекомендуется для большинства случаев — асимметричный, устойчивый к шуму)
    baseline_iarpls, _ = baseline_fitter.iarpls(y, 1e7)
    corrected_iarpls = y - baseline_iarpls

    # Метод 2: aspls (альтернатива, хорошо работает при сильном шуме)
    baseline_aspls, _ = baseline_fitter.aspls(y, lam=1e5)
    corrected_aspls = y - baseline_aspls
   


    # Метод 3: modpoly (полиномиальная аппроксимация, если фон гладкий)
    baseline_modpoly, _ = baseline_fitter.modpoly(y, poly_order=1)
    corrected_modpoly = y - baseline_modpoly
    

    baseline_psalsa, params = baseline_fitter.psalsa(y, 1e5, k=0.05)
    corrected_psalsa = y - baseline_psalsa
    df['dR110'] = corrected_psalsa
    plt.figure('Correct baseline psalsa') 
    df.loc[:,'dR110'].plot(color='k')
    plt.grid(True)
    df['dR110'] = corrected_aspls
    plt.figure('Correct baseline aspls') 
    df.loc[:,'dR110'].plot(color='k')
    plt.grid(True)
    df['dR110'] = corrected_modpoly
    plt.figure('Correct baseline modpoly') 
    df.loc[:,'dR110'].plot(color='k')
    plt.grid(True)

    # Сохранение лучшего результата в DataFrame
    df['dR110'] = corrected_iarpls  # или corrected_aspls / corrected_modpoly

    plt.figure('Correct baseline iarpls') 
    df.loc[:,'dR110'].plot(color='k')
    plt.grid(True)


    signal = df['dR110'].values
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
