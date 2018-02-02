import numpy as np
import xlrd
import matplotlib.pyplot as plt

def sigmoid(inX, param):
	return 1.0 / (1+np.exp(-param*inX))

def isin(origin, filter, invert=False):
	# print "origin", origin
	# print "filter", filter
	ret_lst = []
	if invert == True:
		# if filter element is in origin, return false; else, return true
		for ele in origin:
			if ele in filter:
				ret_lst.append(False)
				ind = filter.index(ele)
				del filter[ind]
			else:
				ret_lst.append(True)
	elif invert == False:
		# if filter element is in origin, return true; else, return false
		for ele in origin:
			if ele in filter:
				ret_lst.append(True)
				ind = filter.index(ele)
				del filter[ind]
			else:
				ret_lst.append(False)

	return ret_lst

			

# def ravel(ele):
# 	print("ele=", ele.shape)

def MergeAndCount(lst, left, mid, right, temp):
	i = left
	j = mid + 1
	k = left
	cnt = 0

	while i <= mid and j <= right:
		if lst[i] <= lst[j]:
			temp[k] = lst[i]
			k = k + 1
			i = i + 1
		else:
			temp[k] = lst[j]
			k = k + 1
			j = j + 1
			cnt += mid -i + 1

	while i <= mid:
		temp[k] = lst[i]
		k = k + 1
		i = i + 1
	while j <= right:
		temp[k] = lst[j]
		k = k + 1
		j = j + 1

	for i in range(left, right + 1):
		lst[i] = temp[i]

	return cnt


def MergeSortAndCount(lst, left, right, temp):
	if (left >= right):
		return 0

	mid = int((left + right) / 2)
	InversionCnt1 = MergeSortAndCount(lst, left, mid, temp)
	InversionCnt2 = MergeSortAndCount(lst, mid+1, right, temp)
	MergeInversionCnt = MergeAndCount(lst, left, mid, right, temp)

	return InversionCnt1 + InversionCnt2 + MergeInversionCnt


def CountInversions(lst, type):
	# pass
	if type == "decrease":
		lst.reverse()
	temp = [0]*len(lst)
	return MergeSortAndCount(lst, 0, len(lst)-1, temp)


def CalDifference(lst):
	'''calculate the difference between neighbors'''
	difference = 0
	for i in range(len(lst)-1):
		difference = difference + np.abs(lst[i][0] - lst[i+1][0]) + np.abs(lst[i][1] - lst[i+1][1])
	return difference


def load_nutrient_data(filename):
    nutrient_book = xlrd.open_workbook(filename)
    nutrient_data = nutrient_book.sheets()[0]
    first_row_name = nutrient_data.row_values(0)
    nutrient_dict = { ind: [] for ind in range(2, len(first_row_name)) }
    name_lst = []
    for i in range(1, nutrient_data.nrows):
        for value, ind in zip(nutrient_data.row_values(i)[2:], range(2, len(nutrient_data.row_values(i)))):
            nutrient_dict[ind].append(value)
        name_lst.append(nutrient_data.row_values(i)[0])
    return first_row_name, nutrient_dict, name_lst


def combine_data(col_name1, col_name2, col_name_lst, nutrient_dict, plot='True'):
    ret_lst = []
    # print(nutrient_dict)
    for l1,l2 in zip(nutrient_dict[col_name_lst.index(col_name1)], nutrient_dict[col_name_lst.index(col_name2)]):
        ret_lst.append([l1, l2])
    if plot == 'True':
        plt.figure(1)
        plt.title("Nutrient data")
        plt.xlabel("items")
        plt.ylabel(col_name1+'/'+col_name2)
        plt.plot(range(1, len(ret_lst)+1), [ret_lst[i][0] for i in range(len(ret_lst))], 'r-', label=col_name1, marker='*')
        plt.plot(range(1, len(ret_lst)+1), [ret_lst[i][1] for i in range(len(ret_lst))], 'b--', label=col_name2, marker='*')
        plt.show()

    return ret_lst





