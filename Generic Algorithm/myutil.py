import numpy as np

def sigmoid(inX):
	return 1.0 / (1+np.exp(-inX))

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
