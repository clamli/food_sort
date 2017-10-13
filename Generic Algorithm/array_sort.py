def VectorInput():
	num = int(input("Number of vectors: "))
	taste_health_list = []
	for i in range(num):
		taste_health_list.append(input("<Taste, Health> %d: "%(i+1)))
	print(taste_health_list)
	return taste_health_list

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
	# print(lst)
	if (left >= right):
		return 0

	mid = int((left + right) / 2)
	InversionCnt1 = MergeSortAndCount(lst, left, mid, temp)
	InversionCnt2 = MergeSortAndCount(lst, mid+1, right, temp)
	MergeInversionCnt = MergeAndCount(lst, left, mid, right, temp)

	return InversionCnt1 + InversionCnt2 + MergeInversionCnt


def CountInversions(lst, type):
	if type == "decrease":
		lst.reverse()
	temp = [0]*len(lst)
	cnt = MergeSortAndCount(lst, 0, len(lst)-1, temp)
	return cnt


def CostFunction(midvec, leftlst, rightlst, theta1=1.0, theta2=1.0, theta3=1.0):
	left_taste_list, left_health_list = [leftlst[i][0] for i in range(len(leftlst))], [leftlst[i][1] for i in range(len(leftlst))]
	right_taste_list, right_health_list = [rightlst[i][0] for i in range(len(rightlst))], [rightlst[i][1] for i in range(len(rightlst))]

	# left_taste_revcnt = CountRevPair(leftlst, "decrease");
	print("-----", left_health_list)
	left_health_revcnt = CountInversions(left_health_list, "increase")
	right_taste_revcnt = CountInversions(right_taste_list, "decrease")
	mid_ifluen = midvec[0] + midvec[1]
	# right_health_revcnt = CountRevPair(rightlst, "increase");

	cost = theta1*(left_health_revcnt+right_taste_revcnt) - theta2*len(leftlst)*len(rightlst) - theta3*mid_ifluen

	return cost


def FindOptimSequence(taste_health_list):
	min_cost = 0
	cur_left_list = []
	cur_right_list = []
	cur_mid_vec = []
	for i in range(len(taste_health_list)):
		tmp_list = taste_health_list[:]
		midvec = tmp_list[i]
		del tmp_list[i]
		leftlst = []
		rightlst = []
		for v in tmp_list:
			if v[1] > midvec[1]:
				rightlst.append(v)
			elif v[0] > midvec[0]:
				leftlst.append(v)
			elif v[1] < midvec[1]:
				leftlst.append(v)
			elif v[0] < midvec[0]:
				rightlst.append(v)
			elif v[0] == midvec[0] or v[1] == midvec[1]:
				if len(rightlst) > len(leftlst):
					leftlst.append(v)
				else:
					rightlst.append(v)

		# print "right list %d:"%i, rightlst
		# print "left list %d:"%i, leftlst, "\n"
		cost = CostFunction(midvec, leftlst, rightlst)
		if cost < min_cost:
			min_cost = cost
			cur_left_list = leftlst
			cur_right_list = rightlst
			cur_mid_vec = midvec


	cur_left_list.sort(lambda x,y:cmp(x[0],y[0]), reverse=True)
	cur_right_list.sort(lambda x,y:cmp(x[1],y[1]))
	return cur_left_list + [cur_mid_vec] + cur_right_list, cur_mid_vec, min_cost

