def append_to_list(a_list):
    for i in range(10):
        a_list.append(i)
    return a_list

a_list = []
a_list = append_to_list(a_list)
print(a_list)