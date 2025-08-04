def find_Bridge(poly_P, poly_Q):
    check_polygon(poly_P)
    check_polygon(poly_Q)   
    case = which_Case(poly_P, poly_Q)
    if (case == 1):
        case_1_result = optimal_type_1_bridge(poly_P, poly_Q)
        case_1_bridge_price = case_1_result[0] 
        case_1_p = case_1_result[1]
        case_1_q = case_1_result[2]
        print("CASE 1:")
        print("bridge price:", case_1_bridge_price)
        print("p:", case_1_p) 
        print("q:", case_1_q)
        print("\n")
    if (case == 2):        
        case_2_result = optimal_type_2_bridge(poly_P, poly_Q)
        case_2_bridge_price = case_2_result[0]
        case_2_p = case_2_result[1]
        case_2_q = case_2_result[2]
        case_2_r = case_2_result[3]
        print("CASE 2:")
        print("bridge price:", case_2_bridge_price)
        print("p:", case_2_p) 
        print("q:", case_2_q)
        print("r:", case_2_r)
        print("\n")

def check_polygon(poly):
    if len(poly) < 3:
        print("Error: Polygon must have at least three vertices.")
        return
    for i in range(len(poly)-1):
        if (poly[i][0] == poly[i+1][0]) or (poly[i][1] == poly[i+1][1]): 
            if not ((poly[i][0] == poly[i+1][0]) and (poly[i][1] == poly[i+1][1])):  
                continue    
        print(f"Error: Vertices {poly[i]} and {poly[i+1]} do not satisfy the condition.")
        return
    if (poly[0][0] == poly[-1][0]) or (poly[0][1] == poly[-1][1]):  
        if not ((poly[0][0] == poly[-1][0]) or (poly[0][1] == poly[-1][1])):
            print(f"Error: Vertices {poly[0]} and {poly[-1]} do not satisfy the condition.")
            return

def which_Case(poly_P, poly_Q):
    case = 1
    min_y_P = min(p[1] for p in poly_P)
    max_y_P = max(p[1] for p in poly_P)
    min_y_Q = min(p[1] for p in poly_Q)
    max_y_Q = max(p[1] for p in poly_Q)
    between_y = 0
    min_x_P = min(p[0] for p in poly_P)
    max_x_P = max(p[0] for p in poly_P)
    min_x_Q = min(p[0] for p in poly_Q)
    max_x_Q = max(p[0] for p in poly_Q)   
    between_x = 0
    condition_1_y = (min_y_P <= min_y_Q <= max_y_P)
    condition_2_y = (min_y_P <= max_y_Q <= max_y_P)
    condition_3_y = (min_y_Q <= min_y_P <= max_y_Q)
    condition_4_y = (min_y_Q <= max_y_P <= max_y_Q)
    if (condition_1_y) or (condition_2_y) or (condition_3_y) or (condition_4_y):        
        between_y = 1 
    condition_1_x = (min_x_P <= min_x_Q <= max_x_P)
    condition_2_x = (min_x_P <= max_x_Q <= max_x_P)
    condition_3_x = (min_x_Q <= min_x_P <= max_x_Q)
    condition_4_x = (min_x_Q <= max_x_P <= max_x_Q)
    if (condition_1_x) or (condition_2_x) or (condition_3_x) or (condition_4_x):   
        between_x = 1
    if (between_y) == 0 and (between_x) == 0:        
        case = 2
    return case

def optimal_type_2_bridge(poly_P, poly_Q):
    transition_points_P = find_transition_points(poly_P)
    transition_points_Q = find_transition_points(poly_Q)    
    L1_projection_points_P = get_L1_projection_points(poly_P)
    L1_projection_points_Q = get_L1_projection_points(poly_Q)
    horizontal_line = find_horizontal_line(poly_P, poly_Q)
    vertical_line = find_vertical_line(poly_P, poly_Q)   
    l1_set_P = find_l1_set(poly_P, transition_points_P, L1_projection_points_P, horizontal_line)
    l2_set_P = find_l2_set(poly_P, transition_points_P, L1_projection_points_P, vertical_line)
    l1_set_Q = find_l1_set(poly_Q, transition_points_Q, L1_projection_points_Q, horizontal_line)
    l2_set_Q = find_l2_set(poly_Q, transition_points_Q, L1_projection_points_Q, vertical_line)
    mu_p1_min = generate_candidates(l1_set_P, poly_P, horizontal_line, vertical_line)  
    mu_p2_min = generate_candidates(l2_set_P, poly_P, horizontal_line, vertical_line)  
    mu_q1_min = generate_candidates(l1_set_Q, poly_Q, horizontal_line, vertical_line)
    mu_q2_min = generate_candidates(l2_set_Q, poly_Q, horizontal_line, vertical_line)
    result = find_F2(mu_p1_min, mu_p2_min, mu_q1_min, mu_q2_min)
    return result
    
def optimal_type_1_bridge(poly_P, poly_Q):
    Bridge_Alg1 = []
    transition_points_P = find_transition_points(poly_P)
    transition_points_Q = find_transition_points(poly_Q)
    L1_projection_points_P = get_L1_projection_points(poly_P)
    L1_projection_points_Q = get_L1_projection_points(poly_Q)
    connectable_points_P = find_add_connectable_points(poly_P, poly_Q, transition_points_P, L1_projection_points_P)
    Bridge_Alg1.extend(connectable_points_P)   
    connectable_points_Q = find_add_connectable_points(poly_Q, poly_P, transition_points_Q, L1_projection_points_Q)   
    Bridge_Alg1.extend(connectable_points_Q)    
    Bridge_Alg1 = remove_duplicates(Bridge_Alg1)
    F1_list_P = find_F1_for_list(poly_P, connectable_points_P)
    F1_list_Q = find_F1_for_list(poly_Q, connectable_points_Q)
    result = F1_min(F1_list_P, F1_list_Q, connectable_points_P, connectable_points_Q)    
    return result

def find_transition_points(polygon):
    sorted_left_vertices = find_farthest_left_vertices(polygon)   
    left_below = sorted_left_vertices[0]
    left_up = sorted_left_vertices[1]
    sorted_right_vertices = find_farthest_right_vertices(polygon)   
    right_below = sorted_right_vertices[0]
    right_up = sorted_right_vertices[1]
    total_distance_left_to_right = find_total_distance(polygon, left_up, right_up)
    total_distance_top_to_bottom = find_total_distance(polygon, right_up, right_below)
    total_distance_right_to_left = find_total_distance(polygon, right_below, left_below)
    total_distance_bottom_to_top = find_total_distance(polygon, left_below, left_up)
    left_point = find_point_for_half_distance(polygon,left_below, total_distance_bottom_to_top)
    right_point = find_point_for_half_distance(polygon, right_up, total_distance_top_to_bottom)
    top_point = find_point_for_half_distance(polygon, left_up, total_distance_left_to_right)
    bottom_point = find_point_for_half_distance(polygon, right_below, total_distance_right_to_left)
    return [top_point, right_point, bottom_point, left_point]

def find_farthest_left_vertices(polygon):
    x_coordinate = lambda point: point[0]
    sorted_polygon_left = sorted(polygon, key=x_coordinate)
    left_vertices = sorted_polygon_left[:2]
    y_coordinate = lambda point: point[1]
    sorted_left_vertices = sorted(left_vertices, key=y_coordinate)
    left_below = sorted_left_vertices[0]
    left_up = sorted_left_vertices[1]
    return left_below, left_up

def find_farthest_right_vertices(polygon):
    y_coordinate = lambda point: point[0]
    sorted_polygon_right = sorted(polygon, key=y_coordinate, reverse=True)
    right_vertices = sorted_polygon_right[:2]
    y_coordinate = lambda point: point[1]
    sorted_right_vertices = sorted(right_vertices, key=y_coordinate)
    right_below = sorted_right_vertices[0]
    right_up = sorted_right_vertices[1]
    return right_below, right_up

def find_total_distance(polygon, start_vertex, end_vertex):
    n = len(polygon)
    total_distance = 0
    start_index = polygon.index(start_vertex)
    end_index = polygon.index(end_vertex)
    current_index = start_index
    while (current_index) != (end_index): 
        x1 = polygon[current_index][0]
        y1 = polygon[current_index][1]
        x2 = polygon[(current_index + 1) % n][0]        # % n to stay in range
        y2 = polygon[(current_index + 1) % n][1]
        distance_x = abs(x1 - x2)
        distance_y = abs(y1 - y2)
        total_distance += distance_x + distance_y
        current_index = (current_index + 1) % n
    return total_distance

def find_point_for_half_distance(polygon, start_point, total_distance):
    n = len(polygon)
    current_distance = 0
    half_distance = total_distance / 2
    for i in range(n): 
        x1 = polygon[i][0]
        y1 = polygon[i][1]
        x2 = polygon[(i + 1) % n][0]
        y2 = polygon[(i + 1) % n][1]
        if (x1, y1) == (start_point):
            for j in range(i,n):     
                x1 = polygon[j][0]
                y1 = polygon[j][1]
                x2 = polygon[(j + 1) % n][0]
                y2 = polygon[(j + 1) % n][1]           
                distance_x = abs(x1 - x2)
                distance_y = abs(y1 - y2)
                edge_distance = distance_x + distance_y
                if (current_distance + edge_distance) >= (half_distance):                                        
                    remaining_distance = half_distance - current_distance  
                    if (x1) == (x2):
                        if (y1) > (y2):
                            transition_point = (x1, y1 - remaining_distance)
                        else:
                            transition_point = (x1, y1 + remaining_distance)                        
                    else:
                        if (x1) > (x2):
                            transition_point = (x1 - remaining_distance, y1)
                        else:
                            transition_point = (x1 + remaining_distance, y1)
                    return transition_point
                current_distance += edge_distance

def get_L1_projection_points(poly):
    n = len(poly)
    projection_points = []
    for i in range(n):
        current_vertex = poly[i]
        x_current = current_vertex[0]
        y_current = current_vertex[1]
        closest_projection = None
        for j in range(n):
            first_next_index = (j + 1) % n
            first_other_vertex = poly[first_next_index]
            first_x_other = first_other_vertex[0]
            first_y_other = first_other_vertex[1]
            second_next_index = (j + 2) % n
            second_other_vertex = poly[second_next_index]
            second_x_other = second_other_vertex[0]
            second_y_other = second_other_vertex[1]
            is_same_x_axis = (first_x_other == second_x_other)
            is_y_current_between = (first_y_other > y_current and second_y_other < y_current)
            is_y_current_between_reverse = (first_y_other < y_current and second_y_other > y_current)
            if (is_same_x_axis) and ((is_y_current_between) or (is_y_current_between_reverse)):  
                projection = (first_x_other, y_current)
                distance_condition = lambda cv, p, cp: (euclidean_distance(cv, p)) < (euclidean_distance(cv, cp))   
                if (closest_projection is None) or (distance_condition(current_vertex, projection, closest_projection)):
                    closest_projection = projection
            is_same_y_axis = (first_y_other == second_y_other)
            is_x_current_between = (first_x_other > x_current > second_x_other)
            is_x_current_between_reverse = (first_x_other < x_current < second_x_other)
            if (is_same_y_axis) and ((is_x_current_between) or (is_x_current_between_reverse)):
                projection = (x_current, first_y_other)
                distance_condition = lambda cv, p, cp: euclidean_distance(cv, p) < euclidean_distance(cv, cp)
                if (closest_projection is None) or (distance_condition(current_vertex, projection, closest_projection)):
                    closest_projection = projection
        if (closest_projection is not None) and (closest_projection not in projection_points):
            projection_points.append(closest_projection)
    return projection_points
    
def find_horizontal_line(poly_P, poly_Q):
    bottom_vertex_P = min(poly_P, key=lambda p: p[1])
    top_vertex_P = max(poly_P, key=lambda p: p[1])
    bottom_vertex_Q = min(poly_Q, key=lambda q: q[1])
    top_vertex_Q = max(poly_Q, key=lambda q: q[1])
    if (bottom_vertex_P[1]) < (bottom_vertex_Q[1]):
        upper_polygon = poly_Q
        lower_polygon = poly_P
        y_coordinate = (bottom_vertex_Q[1] - top_vertex_P[1]) / 2 + top_vertex_P[1]
    else:
        upper_polygon = poly_P
        lower_polygon = poly_Q
        y_coordinate = (bottom_vertex_P[1] - top_vertex_Q[1]) / 2 + top_vertex_Q[1]
    min_x_upper = min(p[0] for p in upper_polygon)
    min_x_lower = min(p[0] for p in lower_polygon)
    max_x_upper = max(p[0] for p in upper_polygon)
    max_x_lower = max(p[0] for p in lower_polygon)
    left_point = min(min_x_upper, min_x_lower)
    right_point = max(max_x_upper, max_x_lower)
    horizontal_line = [(left_point-1, y_coordinate), (right_point+1, y_coordinate)]
    return horizontal_line

def find_vertical_line(poly_P, poly_Q):
    rightmost_vertex_P = max(poly_P, key=lambda p: p[0])
    leftmost_vertex_P = min(poly_P, key=lambda p: p[0])
    rightmost_vertex_Q = max(poly_Q, key=lambda q: q[0])
    leftmost_vertex_Q = min(poly_Q, key=lambda q: q[0])
    if rightmost_vertex_P[0] < leftmost_vertex_Q[0]:
        left_polygon = poly_P
        right_polygon = poly_Q
        x_distance = (leftmost_vertex_Q[0] - rightmost_vertex_P[0]) / 2
        x_coordinate = x_distance + rightmost_vertex_P[0] 
    else:
        left_polygon = poly_Q
        right_polygon = poly_P
        y_distance = (leftmost_vertex_P[0] - rightmost_vertex_Q[0]) / 2
        x_coordinate = y_distance + leftmost_vertex_P[0]
    max_y_left_polygon = max(p[1] for p in left_polygon)
    min_y_left_polygon = min(p[1] for p in left_polygon)
    max_y_right_polygon = max(p[1] for p in right_polygon)
    min_y_right_polygon = min(p[1] for p in right_polygon)
    topmost_point = max(max_y_left_polygon, max_y_right_polygon)
    bottommost_point = min(min_y_left_polygon, min_y_right_polygon)
    top_point = topmost_point + 1
    bottom_point = bottommost_point - 1
    vertical_line = [(x_coordinate, top_point), (x_coordinate, bottom_point)]
    return vertical_line

def find_l1_set(poly, transition_points, L1_projection_points, horizontal_line):
    l1_set = set()
    for vertex in poly:
        l1_set.add(vertex)
    for point in transition_points:
        l1_set.add(point)
    for projection_point in L1_projection_points:
        l1_set.add(projection_point)
    l1_set.difference_update({None})
    if (min(p[1] for p in poly)) < (horizontal_line[0][1]):
        tail_poly = poly[1:]
        first_vertex = poly[0]
        extended_poly = tail_poly + [first_vertex]
        paired_vertices = zip(poly, extended_poly)
        for v1, v2 in paired_vertices:
            if (v1[1] == v2[1]):
                excluded_vertices_condition = lambda v3: (v3[1] < v1[1]) and ( (v1[0] >= v3[0] >= v2[0]) or (v1[0] <= v3[0] <= v2[0]) )
                excluded_vertices = {v3 for v3 in l1_set if excluded_vertices_condition(v3)}
                l1_set -= excluded_vertices
    elif (max(p[1] for p in poly) > horizontal_line[0][1]): 
        tail_poly = poly[1:]
        first_vertex = poly[0]
        extended_poly = tail_poly + [first_vertex]
        paired_vertices = zip(poly, extended_poly)
        for v1, v2 in paired_vertices:
            if (v1[1] == v2[1]):
                excluded_vertices_condition = lambda v3: (v3[1] > v1[1]) and ( (v1[0] >= v3[0] >= v2[0]) or (v1[0] <= v3[0] <= v2[0]) )
                excluded_vertices = {v3 for v3 in l1_set if excluded_vertices_condition(v3)}        
                l1_set -= excluded_vertices
    return l1_set

def find_l2_set(poly, transition_points, L1_projection_points, vertical_line):
    l2_set = set()
    for vertex in poly:
        l2_set.add(vertex)
    for point in transition_points:
        l2_set.add(point)
    for projection_point in L1_projection_points:
        l2_set.add(projection_point)
    l2_set.difference_update({None})
    if (max(p[0] for p in poly) > vertical_line[0][0]):
        tail_poly = poly[1:]
        first_vertex = poly[0]
        extended_poly = tail_poly + [first_vertex]
        paired_vertices = zip(poly, extended_poly)
        for v1, v2 in paired_vertices:
            if (v1[0] == v2[0]):  
               excluded_vertices_condition = lambda v3: (v3[0] > v1[0]) and ( (v1[1] >= v3[1] >= v2[1]) or (v1[1] <= v3[1] <= v2[1]) )
               excluded_vertices = {v3 for v3 in l2_set if excluded_vertices_condition(v3)}
               l2_set -= excluded_vertices
    elif (min(p[0] for p in poly) < vertical_line[0][0]):
        tail_poly = poly[1:]
        first_vertex = poly[0]
        extended_poly = tail_poly + [first_vertex]
        paired_vertices = zip(poly, extended_poly)
        for v1, v2 in paired_vertices:
            if (v1[0] == v2[0]):
                excluded_vertices_condition = lambda v3: (v3[0] < v1[0]) and ( (v1[1] >= v3[1] >= v2[1]) or (v1[1] <= v3[1] <= v2[1]) )
                excluded_vertices = {v3 for v3 in l2_set if excluded_vertices_condition(v3)}
                l2_set -= excluded_vertices
    return l2_set

def generate_candidates(point_set, neighbor_set, horizontal_line, vertical_line):
    min_price = float('inf') 
    min_price_point = None 
    for point in point_set:
        distance_horizontal = abs(point[1] - horizontal_line[0][1])
        distance_vertical = abs(point[0] - vertical_line[0][0])
        max_path = find_most_distant_neighbor_path(point, neighbor_set)
        total_distance = distance_horizontal + distance_vertical + max_path
        if (total_distance < min_price):
            min_price = total_distance
            min_price_point = point
    return min_price_point, min_price

def find_most_distant_neighbor_path(point, neighbor_set):
    sorted_left_vertices = sorted(neighbor_set, key=lambda point: point[0])
    left_vertices = sorted_left_vertices[:2]
    sorted_left_vertices_by_y = sorted(left_vertices, key=lambda point: point[1])
    left_below = sorted_left_vertices_by_y[0] 
    left_up = sorted_left_vertices_by_y[1]
    sorted_right_vertices = sorted(neighbor_set, key=lambda point: point[0], reverse=True)
    right_vertices = sorted_right_vertices[:2]
    sorted_right_vertices_by_y = sorted(right_vertices, key=lambda point: point[1])
    right_below = sorted_right_vertices_by_y[0] 
    right_up = sorted_right_vertices_by_y[1]
    total_edge_length = 0
    tail_neighbor_set = neighbor_set[1:]
    first_vertex = neighbor_set[0]
    extended_neighbor_set = tail_neighbor_set + [first_vertex]
    paired_vertices = zip(neighbor_set, extended_neighbor_set)
    countVertices=0
    FoundPoint=0
    missedVertices=0
    countDistant=0
    foundFirst=0 
    distanceToFirst=0
    distanceToSecond=0
    distanceToThird=0
    distanceToFourth=0        
    second_distanceToFirst=0
    second_distanceToSecond=0
    second_distanceToThird=0
    second_distanceToFourth=0        
    max_path=0
    max_distanceToFirst=0
    max_distanceToSecond=0
    max_distanceToThird=0
    max_distanceToFourth=0 
    for v1, v2 in paired_vertices:
        countVertices += 1
        if (FoundPoint == 0):
            if (v1[1] == v2[1]):  
                condition_1 = (v1[0] <= point[0] <= v2[0])
                condition_2 = (v2[0] <= point[0] <= v1[0])
                if (condition_1) or (condition_2):
                    total_edge_length += abs(point[0] - v1[0])
            elif (v1[0] == v2[0]): 
                condition_1 = (v1[1] <= point[1] <= v2[1])
                condition_2 = (v2[1] <= point[1] <= v1[1])
                if (condition_1) or (condition_2):
                    total_edge_length += abs(point[1] - v1[1])
            if (v1[0] == v2[0]):
                if (point[0] == v1[0]):
                    condition_1 = (v1[1] < point[1] < v2[1])
                    condition_2 = (v2[1] < point[1] < v1[1])                       
                    if (condition_1) or (condition_2): 
                        FoundPoint=1
                        missedVertices = countVertices                            
            if (v1[1] == v2[1]):                    
                if (point[1] == v1[1]):
                    condition_1 = (v1[0] < point[0] < v2[0])
                    condition_2 = (v2[0] < point[1] < v1[0])
                    if (condition_1) or (condition_2):
                        FoundPoint=1
                        missedVertices = countVertices             
        if (FoundPoint == 1):            
            if (v1[1] == v2[1]):  
                condition_1 = (v1[0] <= point[0] <= v2[0])
                condition_2 = (v2[0] <= point[0] <= v1[0])
                if (condition_1) or (condition_2):
                    total_edge_length += abs(point[0] - v1[0])                        
                if (v1==left_up):
                    countDistant+=1
                    distanceToFirst=total_edge_length                        
                    if (foundFirst!=0):
                        foundFirst = 1                    
                if (v1==right_up):
                    countDistant+=1
                    distanceToSecond=total_edge_length                   
                    if (foundFirst!=0):
                        foundFirst = 2                    
                if (v1==right_below):
                    countDistant+=1
                    distanceToThird=total_edge_length                    
                    if (foundFirst!=0):
                        foundFirst = 3                    
                if (v1==left_below):
                    countDistant+=1
                    distanceToFourth=total_edge_length                        
                    if (foundFirst!=0):
                        foundFirst = 4    
            elif (v1[0] == v2[0]):
                condition_1 = (v1[1] <= point[1] <= v2[1])
                condition_2 = (v2[1] <= point[1] <= v1[1])
                if (condition_1) or (condition_2):
                    total_edge_length += abs(point[1] - v1[1])        
                if (v1==left_up):
                    countDistant+=1
                    distanceToFirst=total_edge_length                        
                    if (foundFirst!=0):
                        foundFirst = 1                    
                if (v1==right_up):
                    countDistant+=1
                    distanceToSecond=total_edge_length                        
                    if (foundFirst!=0):
                        foundFirst = 2                    
                if (v1==right_below):
                    countDistant+=1
                    distanceToThird=total_edge_length                       
                    if (foundFirst!=0): 
                        foundFirst = 3                    
                if (v1==left_below):
                    countDistant+=1
                    distanceToFourth=total_edge_length                        
                    if (foundFirst!=0):
                        foundFirst = 4
        if (countDistant < 4):
            tail_neighbor_set2 = neighbor_set[missedVertices:]
            first_vertex2 = neighbor_set[countDistant]
            extended_neighbor_set2 = tail_neighbor_set2 + [first_vertex2]
            paired_vertices2 = zip(neighbor_set, extended_neighbor_set2)
            for v1, v2 in paired_vertices2:   
                if (v1[1] == v2[1]):
                    condition_1 = (v1[0] <= point[0] <= v2[0])
                    condition_2 = (v2[0] <= point[0] <= v1[0])
                    if (condition_1) or (condition_2):
                        total_edge_length += abs(point[0] - v1[0])                            
                    if (v1==left_up):
                        countDistant+=1
                        distanceToFirst=total_edge_length                        
                        if (foundFirst!=0): 
                            foundFirst = 1                        
                    if (v1==right_up):
                        countDistant+=1
                        distanceToSecond=total_edge_length                        
                        if (foundFirst!=0):  
                            foundFirst = 2                        
                    if (v1==right_below):
                        countDistant+=1
                        distanceToThird=total_edge_length                        
                        if (foundFirst!=0):
                            foundFirst = 3                        
                    if (v1==left_below):
                        countDistant+=1
                        distanceToFourth=total_edge_length                        
                        if (foundFirst!=0):
                            foundFirst = 4        
                elif (v1[0] == v2[0]):
                    condition_1 = (v1[1] <= point[1] <= v2[1])
                    condition_2 = (v2[1] <= point[1] <= v1[1])       
                    if (condition_1) or (condition_2):
                        total_edge_length += abs(point[1] - v1[1])            
                    if (v1==left_up):
                        countDistant+=1
                        distanceToFirst=total_edge_length                        
                        if (foundFirst!=0):
                            foundFirst = 1                        
                    if (v1==right_up):
                        countDistant+=1
                        distanceToSecond=total_edge_length                        
                        if (foundFirst!=0):
                            foundFirst = 2                        
                    if (v1==right_below):
                        countDistant+=1
                        distanceToThird=total_edge_length                        
                        if (foundFirst!=0):
                            foundFirst = 3                        
                    if (v1==left_below):
                        countDistant+=1
                        distanceToFourth=total_edge_length                        
                        if (foundFirst!=0):
                            foundFirst = 4    
            if (countDistant == 4):
                break;
    total_edge_length=0
    for v1, v2 in paired_vertices:            
        if (v1[1] == v2[1]):
            condition_1 = (v1[0] <= point[0] <= v2[0])
            condition_2 = (v2[0] <= point[0] <= v1[0])
            if (condition_1) or (condition_2):
                total_edge_length += abs(point[0] - v1[0])           
        elif (v1[0] == v2[0]):
            condition_1 = (v1[1] <= point[1] <= v2[1])
            condition_2 = (v2[1] <= point[1] <= v1[1])   
            if (condition_1) or (condition_2):
                total_edge_length += abs(point[1] - v1[1])
    second_distanceToFirst = total_edge_length - distanceToFirst
    second_distanceToSecond = total_edge_length - distanceToSecond
    second_distanceToThird = total_edge_length - distanceToThird
    second_distanceToFourth = total_edge_length - distanceToFourth
    max_distanceToFirst = max(distanceToFirst, second_distanceToFirst)
    max_distanceToSecond = max(distanceToSecond, second_distanceToSecond)
    max_distanceToThird = max(distanceToThird, second_distanceToThird)
    max_distanceToFourth = max(distanceToFourth, second_distanceToFourth)
    max_path = min(max_distanceToFirst, max_distanceToSecond, max_distanceToThird, max_distanceToFourth)
    return max_path

def find_F2(mu_p1, mu_p2, mu_q1, mu_q2):
    mu_p1_min_price_point = mu_p1[0]
    mu_p1_min_price = mu_p1[1]
    mu_q2_min_price_point = mu_q2[0]
    mu_q2_min_price = mu_q2[1]
    mu_p2_min_price_point = mu_p2[0]
    mu_p2_min_price = mu_p2[1]
    mu_q1_min_price_point = mu_q1[0]
    mu_q1_min_price = mu_q1[1]
    F2_1 = mu_p1_min_price + mu_q2_min_price
    F2_2 = mu_p2_min_price + mu_q1_min_price
    if (F2_1 <= F2_2):
        r = (mu_p1_min_price_point[0], mu_q2_min_price_point[1])
        distance_p1_to_r = euclidean_distance(mu_p1_min_price_point, r)
        distance_r_to_q2 = euclidean_distance(r, mu_q2_min_price_point)
        eucl_result = distance_p1_to_r + distance_r_to_q2
        final_p = mu_p1_min_price_point
        final_q = mu_q2_min_price_point
    else:
        r = (mu_q1_min_price_point[0], mu_p2_min_price_point[1])       
        distance_p2_to_r = euclidean_distance(mu_p2_min_price_point, r)
        distance_r_to_q1 = euclidean_distance(r, mu_q1_min_price_point)
        eucl_result = distance_p2_to_r + distance_r_to_q1
        final_p = mu_q1_min_price_point
        final_q = mu_p2_min_price_point
    result = eucl_result, final_p, final_q, r
    return result

def euclidean_distance(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
    
def find_add_connectable_points(poly_1, poly_2, transition_points, L1_projection_points):   
    connectable_points = set()
    horizontal_points_poly_1 = []
    vertical_points_poly_1 = []
    poly_1_list = list(poly_1)
    paired_points = zip(poly_1_list, poly_1_list[1:])
    horizontal_lines_poly_1 = {(p1, p2) for p1, p2 in paired_points if (p1[1] == p2[1])}
    poly_1_list = list(poly_1)
    paired_points_vertical = zip(poly_1_list, poly_1_list[1:])
    vertical_lines_poly_1 = {(p1, p2) for p1, p2 in paired_points_vertical if (p1[0] == p2[0])}
    poly_1_list = list(poly_1)
    first_point_1 = poly_1_list[0]
    poly_1_list = list(poly_1)
    last_point_1 = poly_1_list[-1]
    vertical_lines_poly_1.add((last_point_1, first_point_1))
    paired_points_horizontal = zip(poly_2, poly_2[1:])
    horizontal_lines_poly_2 = {(p1, p2) for p1, p2 in paired_points_horizontal if (p1[1] == p2[1])}
    paired_points_vertical = zip(poly_2, poly_2[1:])
    vertical_lines_poly_2 = {(p1, p2) for p1, p2 in paired_points_vertical if (p1[0] == p2[0])}    
    first_point_2 = list(poly_2)[0]
    last_point_2 = list(poly_2)[-1]
    vertical_lines_poly_2.add((last_point_2, first_point_2)) 
    for point in horizontal_lines_poly_1:   
        horizontal_points_poly_1.append(point[0])
        horizontal_points_poly_1.append(point[1])
    for point in vertical_lines_poly_1:   
        vertical_points_poly_1.append(point[0])
        vertical_points_poly_1.append(point[1])
    for point in transition_points:   
        x = point[0]
        y = point[1]        
        for (p1, p2) in horizontal_lines_poly_1:           
            x1 = p1[0]
            y1 = p1[1]
            x2 = p2[0]
            y2 = p2[1]
            condition_1 = (y == y1)
            condition_2 = (x1 < x < x2) or (x1 > x > x2)           
            if (condition_1) and (condition_2):
                horizontal_points_poly_1.append(point)
                break                
        for (k1, k2) in vertical_lines_poly_1:                    
            x1 = k1[0]
            y1 = k1[1]
            x2 = k2[0]
            y2 = k2[1]           
            condition_1 = (x == x1)
            condition_2 = (y1 < y < y2) or (y1 > y > y2)            
            if (condition_1) and (condition_2):   
                vertical_points_poly_1.append(point)
                break
    for point in L1_projection_points:        
        x = point[0] 
        y = point[1]       
        for (p1, p2) in horizontal_lines_poly_1:            
            x1 = p1[0]
            y1 = p1[1]
            x2 = p2[0] 
            y2 = p2[1]
            condition_1 = (y == y1)
            condition_2 = (x1 < x < x2) or (x1 > x > x2)            
            if (condition_1) and (condition_2):                   
                horizontal_points_poly_1.append(point)
                break            
        for (k1, k2) in vertical_lines_poly_1:                
            x1 = k1[0] 
            y1 = k1[1]
            x2 = k2[0] 
            y2 = k2[1]                   
            condition_1 = (x == x1)
            condition_2 = (y1 < y < y2) or (y1 > y > y2)            
            if (condition_1) and (condition_2):                       
                vertical_points_poly_1.append(point)
                break   
    for point1 in horizontal_points_poly_1:       
        x1 = point1[0]
        y1 = point1[1]
        check = 0
        matching_pairs = {(p1, p2) for (p1, p2) in horizontal_lines_poly_2 if ((p1[0] <= x1 <= p2[0]) or (p1[0] >= x1 >= p2[0]))}
        for p1, p2 in matching_pairs:            
            check = 0           
            matching_point = (x1, p1[1])
            x_matching = matching_point[0]
            y_matching = matching_point[1]
            for (x_p1, y_p1), (x_p2, y_p2) in horizontal_lines_poly_2:               
                condition_1 = (x_p1 < x_matching < x_p2) or (x_p1 > x_matching > x_p2) or (x_p1 == x_matching) or (x_matching == x_p2) 
                condition_2 = (y1 < y_p1 < y_matching) or (y1 > y_p1 > y_matching)
                if (condition_1) and (condition_2):                   
                    check=1
            for (x_p1, y_p1), (x_p2, y_p2) in horizontal_lines_poly_1:               
                condition_1 = (x_p1 < x_matching < x_p2) or (x_p1 > x_matching > x_p2) or (x_p1 == x_matching) or (x_matching == x_p2)
                condition_2 = (y1 < y_p1 < y_matching) or (y1 > y_p1 > y_matching)
                if (condition_1) and (condition_2):               
                    check=1                   
            if (check==0):
                connectable_points.add((point1, matching_point))
    for point1 in vertical_points_poly_1:        
        x1 = point1[0]
        y1 = point1[1]
        check = 0    
        matching_pairs = {(p1, p2) for (p1, p2) in vertical_lines_poly_2 if (p1[1] <= y1 <= p2[1]) or (p1[1] >= y1 >= p2[1])}   
        for p1, p2 in matching_pairs:                
            matching_point = (p1[0], y1)
            x_matching = matching_point[0]
            y_matching = matching_point[1]    
            for (x_p1, y_p1), (x_p2, y_p2) in vertical_lines_poly_2:                    
                condition_1 = (y_p1 < y_matching < y_p2) or (y_p1 > y_matching > y_p2) or (y_p1 == y_matching) or (y_matching == y_p2)
                condition_2 = (x1 < x_p1 < x_matching) or (x1 > x_p1 > x_matching)   
                if (condition_1) and (condition_2):                       
                    check=1    
            for (x_p1, y_p1), (x_p2, y_p2) in vertical_lines_poly_1:                    
                condition_1 = (y_p1 < y_matching < y_p2) or (y_p1 > y_matching > y_p2) or (y_p1 == y_matching) or (y_matching == y_p2)
                condition_2 = (x1 < x_p1 < x_matching) or (x1 > x_p1 > x_matching)    
                if (condition_1) and (condition_2):                                    
                    check=1                    
            if (check == 0):
                connectable_points.add((point1, matching_point))
    return connectable_points

def find_F1(points_pairs_set, neighbor_set):
    point1 = points_pairs_set[0]
    point2 = points_pairs_set[1]
    F1_list_prices = set()
    distance_horizontal = abs(point1[0] - point2[0])
    distance_vertical = abs(point1[1] - point2[1])
    max_path = find_most_distant_neighbor_path(point1, neighbor_set)
    total_distance = distance_horizontal + distance_vertical + max_path
    F1_list_prices.add(total_distance)
    return F1_list_prices

def remove_duplicates(Bridge_Alg1):
    unique_bridge_set = sort_list(Bridge_Alg1)
    Bridge_Alg1 = sort_list(unique_bridge_set)
    return Bridge_Alg1

def sort_list(list):
    sorted_pairs = []
    for pair in list:
        sorted_pair = tuple(sorted(pair))
        sorted_pairs.append(sorted_pair)
    unique_list_set = set(sorted_pairs)
    return unique_list_set

def find_F1_for_list(poly, connectable_points):
    F1_list = []
    for pair in connectable_points:
        result = find_F1(pair, poly)
        F1_list.append(result)
    return F1_list

def F1_min(F1_list_P, F1_list_Q, connectable_points_P, connectable_points_Q):
    zipped_values = zip(F1_list_P, connectable_points_P)
    min_value_pair = min(zipped_values, key=lambda x: x[0])
    F1_min_P_price = min_value_pair[0]
    F1_min_P_point = min_value_pair[1]
    zipped_values_Q = zip(F1_list_Q, connectable_points_Q)
    min_value_pair_Q = min(zipped_values_Q, key=lambda x: x[0])
    F1_min_Q_price = min_value_pair_Q[0]
    F1_min_Q_point = min_value_pair_Q[1]
    if (F1_min_P_price < F1_min_Q_price):
        F1_min_price = F1_min_P_price
        F1_min_point = F1_min_P_point
    else:
        F1_min_price = F1_min_Q_price
        F1_min_point = F1_min_Q_point
    F1_min_price = F1_min_price.pop()
    result = (F1_min_price, F1_min_point[0], F1_min_point[1])
    return result

# Example usage rectilinear_polygon:

# CASE 1:
poly_P = [(0,5.5), (1,5.5), (1,2), (2,2), (2,3), (3,3), (3,2), (4,2), (4,1), (0,1)]
poly_Q = [(3,6), (4,6), (4,7), (6,7), (6,8), (7,8), (7,5), (6,5), (6,4), (4,4), (4,5), (3,5)]
find_Bridge(poly_P, poly_Q)

# CASE 2:
poly_P = [(10,1), (10,5), (7,5), (7,7), (6,7), (6,1)]
poly_Q = [(9,10), (9,12), (8,12), (8,11), (7,11), (7,10)]
find_Bridge(poly_P, poly_Q)

# CASE 3: 
poly_P = [(0, 0),(0, 5),(2, 5),(2, 3),(6, 3),(6, 4),(7, 4),(7, 2),(8, 2),(8, 0)]
poly_Q = [(10, 7), (10, 8), (12, 8), (12, 10), (14, 10), (14, 8), (15, 8), (15, 7)]
find_Bridge(poly_P, poly_Q)
