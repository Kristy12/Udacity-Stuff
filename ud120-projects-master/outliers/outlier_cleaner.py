#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        clean away the 10% of points that have the largest
        residual errors (different between the prediction
        and the actual net worth)

        return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error)
    """
    
    cleaned_data = []

    ### your code goes here
    error = abs(predictions - net_worths)
    cleaned_data = zip(ages[::,0], net_worths[::,0], error[::,0])
    cleaned_data.sort(key=lambda tup: tup[2])
    index = int(-0.1*len(clean_data)) - 1
    del cleaned_data[index:-1]
    #print clean_data
    return cleaned_data

