import pandas as pd
import os
import pprint

if __name__ == '__main__':
    folder_name = os.getcwd() + '/'
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
                else:
                    df = df.join(sorensen_dice, how='inner')
                print 'Num inst.', len(sorensen_dice)
                print 'Mean', sorensen_dice.mean().values[0]
                print 'STD', sorensen_dice.std().values[0]
                print '\n'

