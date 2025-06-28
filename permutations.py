

def permutations(list_of_lists):
    if len(list_of_lists) == 1:
        for e in list_of_lists[0]:
            yield [e]

        return

    l = list_of_lists[0]
    for e in l:
        for g in permutations(list_of_lists[1:]):
            yield [e] + g


list_of_lists = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
]
for p in permutations(list_of_lists):
    print(p)
