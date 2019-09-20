from Cleaning_Creating_MEWS import*
from Create_test_train import*
from Model import*
from plotting_functons import*

def main():
    print("python main function")

vitals_above_seven = pd.read_csv('vitals_above_seven.csv')
vitals_below_seven = pd.read_csv('vitals_below_seven.csv')

vitals_above_seven_wanted = filter_wanted_data(vitals_above_seven)


if __name__ == '__main__':
    main()
