
def s(n):
    matrix_list = []
    directions_list=[[(0, 1), (1, 0), (0, -1), (-1, 0)], 
                     [(1, 0), (0, -1), (-1, 0), (0, 1)], 
                     [(0, -1), (-1, 0), (0, 1), (1, 0)], 
                     [(-1, 0), (0, 1), (1, 0), (0, -1)],
                     [(0, 1), (-1, 0), (0, -1), (1, 0)],
                     [(0, -1), (1, 0), (0, 1), (-1, 0)],
                     [(1, 0), (0, 1), (-1, 0), (0, -1)],
                     [(-1, 0), (0, -1), (1, 0), (0, 1)]]
    for i in range(len(directions_list)):
        matrix = [[0] * n for _ in range(n)]
        x, y = n // 2, n // 2
        
        directions = directions_list[i]
        current_direction = 0
        steps = 1
        value = 1 
        while value <= n*n:
            for _ in range(2): 
                for _ in range(steps):
                    if 0 <= x < n and 0 <= y < n:
                        matrix[x][y] = value
                        value += 1
                    x += directions[current_direction][0]
                    y += directions[current_direction][1]
                current_direction = (current_direction + 1) % 4
            steps += 1
        rearrange_list = []
        re_rearrange_list = []
        for row in matrix:
            for num in row:
                rearrange_list.append(num-1)
                re_rearrange_list.append(n**2 - num)  
        matrix_list.append(rearrange_list)
        matrix_list.append(re_rearrange_list)
        original_order_indexes_list = []
        for k in range(len(matrix_list)):
            index_mapping = {index: i for i, index in enumerate(matrix_list[k])}
            original_order_indexes = [index_mapping[i] for i in range(len(matrix_list[k]))]
            original_order_indexes_list.append(original_order_indexes)
    return matrix_list, original_order_indexes_list


def z1(n):
    matrix = [[0] * n for _ in range(n)]
    num = 1
    for i in range(n):
        if i % 2 == 0:
            for j in range(n):
                matrix[i][j] = num
                num += 1
        else:
            for j in range(n-1, -1, -1):
                matrix[i][j] = num
                num += 1
    return matrix

def z2(n):
    matrix = [[0] * n for _ in range(n)]
    num = 1
    for j in range(n):
        if j % 2 == 0:
            for i in range(n):
                matrix[i][j] = num
                num += 1
        else:
            for i in range(n-1, -1, -1):
                matrix[i][j] = num
                num += 1
    return matrix

def z3(n):
    matrix = z1(n)
    matrix = [row[::-1] for row in matrix]  
    return matrix

def z4(n):
    matrix = z2(n)
    matrix = [col[::-1] for col in matrix]  
    return matrix

def z5(n):
    matrix = z1(n)
    matrix = matrix[::-1] 
    return matrix

def z6(n):
    matrix = z2(n)
    matrix = matrix[::-1]  
    return matrix

def z7(n):
    matrix = z5(n)
    matrix = [col[::-1] for col in matrix]  
    return matrix

def z8(n):
    matrix = z6(n)
    matrix = [col[::-1] for col in matrix] 
    return matrix

def z(n: int, i: int):
    rearrange_list = []
    if i%8==1:
        matrix = z1(n)
    elif i%8==2:
        matrix = z2(n)
    elif i%8==3:
        matrix = z3(n)
    elif i%8==4:
        matrix = z4(n)
    elif i%8==5:
        matrix = z5(n)
    elif i%8==6:
        matrix = z6(n)
    elif i%8==7:
        matrix = z7(n)
    elif i%8==0:
        matrix = z8(n)
    for row in matrix:
        for num in row:
            rearrange_list.append(num-1)

    index_mapping = {index: i for i, index in enumerate(rearrange_list)}
    original_order_indexes = [index_mapping[i] for i in range(len(rearrange_list))]
    return rearrange_list, original_order_indexes

def v(n: int):
    m_list = []
    m_list.append(z1(n))
    m_list.append(z2(n))
    m_list.append(z7(n))
    m_list.append(z8(n))

    order_list = []
    original_list = []

    for m in m_list:
        rearrange_list = []
        for row in m:
            for num in row:
                rearrange_list.append(num-1)
        order_list.append(rearrange_list)

        index_mapping = {index: i for i, index in enumerate(rearrange_list)}
        original_order_indexes = [index_mapping[i] for i in range(len(rearrange_list))]

        original_list.append(original_order_indexes)

    return order_list, original_list