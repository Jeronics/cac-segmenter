import os

import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':
    results_folder_name = 'segment_results_alpert_new/'
    folder_name =  results_folder_name
    first = None
    for r, s, f in os.walk(folder_name):
        for sub in s:
            print sub
            sorensen_dice_file = folder_name + sub + '/' + 'sorensen_dice_coeff.txt'
            if os.path.exists(sorensen_dice_file):
                sorensen_dice = pd.read_csv(sorensen_dice_file, index_col=0, sep='\t', header=None)
                # pprint.pprint(sorensen_dice),
                # Remove duplicates
                grouped = sorensen_dice.groupby(level=0)
                sorensen_dice = grouped.last()
                sorensen_dice.columns = [sub]
                if not first:
                    df = sorensen_dice
                    first = True
                else:
                    df = df.join(sorensen_dice, how='inner')
                print 'Num inst.', len(sorensen_dice)
                print 'Mean', sorensen_dice.mean().values[0]
                print 'STD', sorensen_dice.std().values[0]
                print '\n'

        df = df.fillna(-1)
        plt.figure()
        for sub in s:
            sorensen_dice_file = folder_name + sub + '/' + 'sorensen_dice_coeff.txt'
            print df.columns
            print df[sub].values
            if os.path.exists(sorensen_dice_file):
                plt.plot(df[sub].values)
                plt.legend(s, loc='lower center')
        plt.show()



