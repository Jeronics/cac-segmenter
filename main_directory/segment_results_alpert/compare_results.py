import pandas as pd
import os
import pprint
if __name__ == '__main__':
    folder_name = os.getcwd()+'/'
    for r, s, f in os.walk(folder_name):
        for sub in s:
            print sub
            sorensen_dice_file = folder_name + sub + '/' + 'sorensen_dice_coeff.txt'
            if os.path.exists(sorensen_dice_file):
                sorensen_dice = pd.read_csv(sorensen_dice_file, index_col=0, sep='\t', header=None)
                # pprint.pprint(sorensen_dice),
                print sorensen_dice.mean(), sorensen_dice.std()