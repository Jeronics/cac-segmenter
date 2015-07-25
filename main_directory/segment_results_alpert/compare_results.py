import pandas as pd
import os

if __name__ == '__main__':
    folder_name = os.getcwd()+'/'
    for r, s, f in os.walk(folder_name):
        for sub in s:
            print sub
            sorensen_dice_file = folder_name + sub + '/' + 'sorensen_dice_coeff.txt'
            sorensen_dice = pd.DataFrame.from_csv(sorensen_dice_file, header=0, sep='\t')
            print sorensen_dice.mean()