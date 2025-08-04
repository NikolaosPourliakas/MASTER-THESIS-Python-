def find_Bridge(poly_P, poly_Q):

    check_polygon(poly_P)
    check_polygon(poly_Q)   

    # recognise between Cases 1 and 2

    case = which_Case(poly_P, poly_Q)

    # Case 1: the polygons are connected by 1 rectilinear line segment 

    if (case == 1):
        
        case_1_result = optimal_type_1_bridge(poly_P, poly_Q)
        
        # bridge price = euclidean(p,q)

        case_1_bridge_price = case_1_result[0] 

        # bridge connects points p and q

        case_1_p = case_1_result[1]
        case_1_q = case_1_result[2]

        print("CASE 1:")
        print("bridge price:", case_1_bridge_price)
        print("p:", case_1_p) 
        print("q:", case_1_q)
        print("\n")

    # Case 2: the polygons are connected by 2 rectilinear line segments 

    if (case == 2):
        
        case_2_result = optimal_type_2_bridge(poly_P, poly_Q)
        
        # bridge price = euclidean(p,r) + euclidean(r,q)

        case_2_bridge_price = case_2_result[0]

        # bridge connects points p,r and r,q

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
    
    # Check if the polygon has at least three vertices
    
    if len(poly) < 3:
        
        print("Error: Polygon must have at least three vertices.")
        return

    # Iterate through the vertices
    
    for i in range(len(poly)-1):
        
        # Check if x or y coordinates are equal, but not both
        
        if (poly[i][0] == poly[i+1][0]) or (poly[i][1] == poly[i+1][1]):
            
            if not ((poly[i][0] == poly[i+1][0]) and (poly[i][1] == poly[i+1][1])):
                
                continue
            
        print(f"Error: Vertices {poly[i]} and {poly[i+1]} do not satisfy the condition.")
        
        return

    # Check the condition for the first and last vertices
    
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

    # poly_P and poly_Q are lists of points representing the polygons

    # STEP 1
    
    '''
    print("poly_P : \n", poly_P, "\n")
    print("poly_Q : \n", poly_Q, "\n")
    '''
    
    # 1.1 : Find Sets T(P), T(Q) of Transition Points

    transition_points_P = find_transition_points(poly_P)
    transition_points_Q = find_transition_points(poly_Q)
    
    '''
    print("transition_points_P : \n", transition_points_P, "\n")
    print("transition_points_Q : \n", transition_points_Q, "\n")
    '''

    # 1.2 : Find Sets L1P(P), L1P(Q) of L1-Projection Points

    L1_projection_points_P = get_L1_projection_points(poly_P)
    L1_projection_points_Q = get_L1_projection_points(poly_Q)

    '''
    print("L1_projection_points_P : \n", L1_projection_points_P, "\n")
    print("L1_projection_points_Q : \n", L1_projection_points_Q, "\n")
    '''

    # STEP 2
    
    # 2.1 : Find lines that are between the polygons

    horizontal_line = find_horizontal_line(poly_P, poly_Q)
    vertical_line = find_vertical_line(poly_P, poly_Q)
    
    '''
    print("horizontal_line : \n", horizontal_line, "\n")
    print("vertical_line : \n", vertical_line, "\n")
    '''

    # 2.2 : Find l1(P), l2(P), l1(Q), l2(Q)

    # the vertices, transition and L1-projection points of the polygon (P/Q), visible by 
    # l1_set_ : the horizontal line , l2_set_ : the vertical line

    l1_set_P = find_l1_set(poly_P, transition_points_P, L1_projection_points_P, horizontal_line)
    l2_set_P = find_l2_set(poly_P, transition_points_P, L1_projection_points_P, vertical_line)

    l1_set_Q = find_l1_set(poly_Q, transition_points_Q, L1_projection_points_Q, horizontal_line)
    l2_set_Q = find_l2_set(poly_Q, transition_points_Q, L1_projection_points_Q, vertical_line)

    '''
    print("l1_set_P \n", l1_set_P, "\n")
    print("l2_set_P \n", l2_set_P, "\n")
    print("l1_set_Q \n", l1_set_Q, "\n")
    print("l2_set_Q \n", l2_set_Q, "\n")
    '''

    # STEP 3 :  
    
    # m(p1) = min[(p belongs in I1) l1(P)] m(p), 
    # m(p2) = min[(p belongs in l2) l2(P)] m(p),
    # m(q1) = min[(q belongs in I1) l1(Q)] m(q), 
    # m(q2) = min[(q belongs in l2) l2(Q)] m(q),

    # m(x) is the mu_x_min_price, but we need the point for STEP 4

    mu_p1_min = generate_candidates(l1_set_P, poly_P, horizontal_line, vertical_line)  
    mu_p2_min = generate_candidates(l2_set_P, poly_P, horizontal_line, vertical_line)
    
    mu_q1_min = generate_candidates(l1_set_Q, poly_Q, horizontal_line, vertical_line)
    mu_q2_min = generate_candidates(l2_set_Q, poly_Q, horizontal_line, vertical_line)

    '''
    print("mu_p1_min \n", mu_p1_min, "\n")
    print("mu_p2_min \n", mu_p2_min, "\n")
    print("mu_q1_min \n", mu_q1_min, "\n")
    print("mu_q2_min \n", mu_q2_min, "\n")
    '''

    # STEP 4

    # Find F2(p1,q2) = m(p1) + m(q2), F2(p2,q1) = m(p2) + m(q1)

    result = find_F2(mu_p1_min, mu_p2_min, mu_q1_min, mu_q2_min)

    '''
    bridge_price = result[0]
    p = result[1]
    q = result[2]
    r = result[3]

    print("result \n", "bridge price:", bridge_price, "\n")
    print("p:", p, "\n", "q:", q, "\n", "r:", r, "\n")
    '''

    return result
    
    
def optimal_type_1_bridge(poly_P, poly_Q):

    # poly_P and poly_Q are lists of points representing the polygons

    # STEP 1 
    
    # Initialize an empty list for points
    
    Bridge_Alg1 = []
    
    '''
    print("poly_P : \n", poly_P, "\n")
    print("poly_Q : \n", poly_Q, "\n")
    '''
    
    # STEP 2
    
    # 2.1 : Find Sets T(P), T(Q) of Transition Points

    transition_points_P = find_transition_points(poly_P)
    transition_points_Q = find_transition_points(poly_Q)
    
    '''
    print("transition_points_P : \n", transition_points_P, "\n")
    print("transition_points_Q : \n", transition_points_Q, "\n")
    '''

    # 2.2 : Find Sets L1P(P), L1P(Q) of L1-Projection Points

    L1_projection_points_P = get_L1_projection_points(poly_P)
    L1_projection_points_Q = get_L1_projection_points(poly_Q)

    '''
    print("L1_projection_points_P : \n", L1_projection_points_P, "\n")
    print("L1_projection_points_Q : \n", L1_projection_points_Q, "\n")
    '''

    # STEP 3
    
    # Call find_add_connectable_points of P and add the result to Bridge_Alg1

    connectable_points_P = find_add_connectable_points(poly_P, poly_Q, transition_points_P, L1_projection_points_P)
    
    Bridge_Alg1.extend(connectable_points_P)
    
    '''
    print("connectable_points_P: \n", connectable_points_P, "\n")
    '''

    # STEP 4

    # Call find_and_add_connectable_points of Q and add the result to Bridge_Alg1

    connectable_points_Q = find_add_connectable_points(poly_Q, poly_P, transition_points_Q, L1_projection_points_Q)
    
    Bridge_Alg1.extend(connectable_points_Q)
    
    '''
    print("connectable_points_Q: \n", connectable_points_Q, "\n")
    '''
    
    # STEP 5

    # remove the duplicates from Bridge_Alg1 for cases like vertice to vertice

    Bridge_Alg1 = remove_duplicates(Bridge_Alg1)

    '''
    print ("Bridge_Alg1: \n", Bridge_Alg1, "\n")
    '''

    # Calculate prices for each pair

    F1_list_P = find_F1_for_list(poly_P, connectable_points_P)
    F1_list_Q = find_F1_for_list(poly_Q, connectable_points_Q)

    # STEP 6

    result = F1_min(F1_list_P, F1_list_Q, connectable_points_P, connectable_points_Q)
    
    return result
    
    

# HELPER FUNCTIONS:


# STEP 1

# STEP 1.1 : Transition points are 4 points, 1 for each side of the polygon (top,below,right,left)
#            such as all the points between them have the same farthest most distant neighbor vertice
#            and we calculate the distance as the length of the rectilinear path connecting them
#            meaning the total length of the edges connecting the vertices between them


def find_transition_points(polygon):

    sorted_left_vertices = find_farthest_left_vertices(polygon)
    
    left_below = sorted_left_vertices[0]
    left_up = sorted_left_vertices[1]

    sorted_right_vertices = find_farthest_right_vertices(polygon)
    
    right_below = sorted_right_vertices[0]
    right_up = sorted_right_vertices[1]

    '''
    print(left_up, left_below, right_up, right_below)
    '''
    
    # Calculate the total distance for each of the 4 pairs of vertices, for the polygon's 4 sides

    total_distance_left_to_right = find_total_distance(polygon, left_up, right_up)
    total_distance_top_to_bottom = find_total_distance(polygon, right_up, right_below)

    total_distance_right_to_left = find_total_distance(polygon, right_below, left_below)
    total_distance_bottom_to_top = find_total_distance(polygon, left_below, left_up)

    # Find the transtion points for each of the 4 pairs of vertices, the half of the total distance

    left_point = find_point_for_half_distance(polygon,left_below, total_distance_bottom_to_top)
    right_point = find_point_for_half_distance(polygon, right_up, total_distance_top_to_bottom)

    top_point = find_point_for_half_distance(polygon, left_up, total_distance_left_to_right)
    bottom_point = find_point_for_half_distance(polygon, right_below, total_distance_right_to_left)

    return [top_point, right_point, bottom_point, left_point]


def find_farthest_left_vertices(polygon):
    
    # Define a lambda function to extract the x-coordinate from each point

    x_coordinate = lambda point: point[0]

    # Sort the polygon based on x-coordinate using the defined lambda function

    sorted_polygon_left = sorted(polygon, key=x_coordinate)

    # Get the first two points from the sorted_polygon_left

    left_vertices = sorted_polygon_left[:2]

    # Define a lambda function to extract the y-coordinate from each point

    y_coordinate = lambda point: point[1]

    # Sort the left_vertices based on y-coordinate and get the two points

    sorted_left_vertices = sorted(left_vertices, key=y_coordinate)

    # Assign the first value to left_below and the second value to left_up

    left_below = sorted_left_vertices[0]
    left_up = sorted_left_vertices[1]

    return left_below, left_up


def find_farthest_right_vertices(polygon):

    # Define a lambda function to extract the y-coordinate from each point

    y_coordinate = lambda point: point[0]

    # Sort the polygon based on y-coordinate in reverse order

    sorted_polygon_right = sorted(polygon, key=y_coordinate, reverse=True)

    # Get the first two points from the sorted_polygon_right

    right_vertices = sorted_polygon_right[:2]

    # Define a lambda function to extract the y-coordinate from each point

    y_coordinate = lambda point: point[1]

    # Sort the right_vertices based on y-coordinate and get the two points

    sorted_right_vertices = sorted(right_vertices, key=y_coordinate)

    # Assign the first value to right_below and the second value to right_up

    right_below = sorted_right_vertices[0]
    right_up = sorted_right_vertices[1]

    return right_below, right_up


# This method calculates the total distance of 
# the combination of edges connecting the pair of vertices

def find_total_distance(polygon, start_vertex, end_vertex):
    
    n = len(polygon)
    total_distance = 0

    # Find the indices of the start and end vertices in the polygon
    # in order to know when to start, and when to stop

    start_index = polygon.index(start_vertex)
    end_index = polygon.index(end_vertex)

    # set the current vertice as the starting vertice

    current_index = start_index
    
    # Check all the vertices from the start to the end of the side

    while (current_index) != (end_index):
        
        # For consecutive vertices, add their edge's distance to the total

        x1 = polygon[current_index][0]
        y1 = polygon[current_index][1]

        x2 = polygon[(current_index + 1) % n][0]        # % n to stay in range
        y2 = polygon[(current_index + 1) % n][1]

        distance_x = abs(x1 - x2)
        distance_y = abs(y1 - y2)

        total_distance += distance_x + distance_y

        current_index = (current_index + 1) % n

    '''
    print(total_distance,x1,y1,x2,y2)
    '''
    
    return total_distance


# This method calculates half of the total distance of the pair of vertices

def find_point_for_half_distance(polygon, start_point, total_distance):
    
    n = len(polygon)
    current_distance = 0
    half_distance = total_distance / 2
    
    # Check every vertice of the polygon, until you find the starting one

    for i in range(n):
        
        # (x1,y1) the first vertice of the pair, and (x2,y2) the second

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

                # If you cover the half distance with the currrent edge, 
                # the transition point is located on it

                if (current_distance + edge_distance) >= (half_distance):
                    
                    remaining_distance = half_distance - current_distance
                    
                    # If the consecutive vertices create a horizontal edge 
                    # calculate the y price, differentiate the upper and lower side

                    if (x1) == (x2):

                        if (y1) > (y2):

                            transition_point = (x1, y1 - remaining_distance)

                        else:
                            transition_point = (x1, y1 + remaining_distance)
                        
                    # If the consecutive vertices create a vertical edge
                    # calculate the x price, differentiate the left and right side

                    else:

                        if (x1) > (x2):

                            transition_point = (x1 - remaining_distance, y1)

                        else:

                            transition_point = (x1 + remaining_distance, y1)
                    
                    return transition_point
    
                # If you haven't covered the half distance with the currrent edge, 
                # add it to the total distance covered, and move to the next edge  

                current_distance += edge_distance


# STEP 1.2 : L1 projection points are the points projected by a vertice on an edge of the polygon

def get_L1_projection_points(poly):

    n = len(poly)
    projection_points = []

    for i in range(n):

        current_vertex = poly[i]

        x_current = current_vertex[0]
        y_current = current_vertex[1]

        # Initialize variables to store the closest pair and its projection point

        closest_projection = None

        # Iterate over pairs of consecutive vertices

        for j in range(n):

            # Set the pair of vertices you compare to the one you check

            first_next_index = (j + 1) % n
            first_other_vertex = poly[first_next_index]

            first_x_other = first_other_vertex[0]
            first_y_other = first_other_vertex[1]

            second_next_index = (j + 2) % n
            second_other_vertex = poly[second_next_index]

            second_x_other = second_other_vertex[0]
            second_y_other = second_other_vertex[1]

            # Check if the conditions are met for x-axis projection

            is_same_x_axis = (first_x_other == second_x_other)

            is_y_current_between = (first_y_other > y_current and second_y_other < y_current)
            is_y_current_between_reverse = (first_y_other < y_current and second_y_other > y_current)

            # if the pair is on a vertical edge, and are the one above and the other below the vertice 
            # add the point projected on the y axis price of the vertice on their edge 

            if (is_same_x_axis) and ((is_y_current_between) or (is_y_current_between_reverse)):
   
                projection = (first_x_other, y_current)

                distance_condition = lambda cv, p, cp: (euclidean_distance(cv, p)) < (euclidean_distance(cv, cp))
                
                if (closest_projection is None) or (distance_condition(current_vertex, projection, closest_projection)):

                    closest_projection = projection

            # Check if the conditions are met for y-axis projection

            is_same_y_axis = (first_y_other == second_y_other)

            is_x_current_between = (first_x_other > x_current > second_x_other)
            is_x_current_between_reverse = (first_x_other < x_current < second_x_other)

            # if the pair is on a horizontal edge, and are the one left and the other right of the vertice 
            # add the point projected on the x-axis price of the vertice on their edge 

            if (is_same_y_axis) and ((is_x_current_between) or (is_x_current_between_reverse)):

                projection = (x_current, first_y_other)

                distance_condition = lambda cv, p, cp: euclidean_distance(cv, p) < euclidean_distance(cv, cp)

                if (closest_projection is None) or (distance_condition(current_vertex, projection, closest_projection)):

                    closest_projection = projection

        # Append the closest projection point to the list if it is not None and not already in the list

        if (closest_projection is not None) and (closest_projection not in projection_points):

            projection_points.append(closest_projection)

    '''
    print(projection_points)
    '''

    return projection_points
    
    
# STEP 2

# STEP 2.1 : The horizontal line is above the lower and below the upper polygon, 
#            so we use the midpoint of the distance between their top and bottom vertice respectively

def find_horizontal_line(poly_P, poly_Q):

    # Find the vertexes with the minimum and the max y-coordinate in poly_P

    bottom_vertex_P = min(poly_P, key=lambda p: p[1])
    top_vertex_P = max(poly_P, key=lambda p: p[1])

    # Find the vertexes with the minimum and the max y-coordinate in poly_Q

    bottom_vertex_Q = min(poly_Q, key=lambda q: q[1])
    top_vertex_Q = max(poly_Q, key=lambda q: q[1])

    # Determine which polygon is upper and which is lower

    if (bottom_vertex_P[1]) < (bottom_vertex_Q[1]):

        upper_polygon = poly_Q
        lower_polygon = poly_P
        
        # Midpoint between bottom of the upper and top of the lower
        
        y_coordinate = (bottom_vertex_Q[1] - top_vertex_P[1]) / 2 + top_vertex_P[1]
        
    else:

        upper_polygon = poly_P
        lower_polygon = poly_Q
        
        # Midpoint between bottom of the upper and top of the lower 

        y_coordinate = (bottom_vertex_P[1] - top_vertex_Q[1]) / 2 + top_vertex_Q[1]

    # Find the farthest left and right points on the x-axis for both polygons

    min_x_upper = min(p[0] for p in upper_polygon)
    min_x_lower = min(p[0] for p in lower_polygon)

    max_x_upper = max(p[0] for p in upper_polygon)
    max_x_lower = max(p[0] for p in lower_polygon)

    left_point = min(min_x_upper, min_x_lower)
    right_point = max(max_x_upper, max_x_lower)

    # Create a horizontal line using the calculated coordinates 
    # (the -1 is added in order to always include the max vertices)

    horizontal_line = [(left_point-1, y_coordinate), (right_point+1, y_coordinate)]

    return horizontal_line


# The vertical line is right of the left and left of the right polygon, 
# so we use the midpoint of the distance between their max left and right vertice respectively

def find_vertical_line(poly_P, poly_Q):

    # Find the vertex with the maximum x-coordinate in poly_P

    rightmost_vertex_P = max(poly_P, key=lambda p: p[0])
    leftmost_vertex_P = min(poly_P, key=lambda p: p[0])

    # Find the vertex with the minimum x-coordinate in poly_Q

    rightmost_vertex_Q = max(poly_Q, key=lambda q: q[0])
    leftmost_vertex_Q = min(poly_Q, key=lambda q: q[0])

    # Determine which polygon is on the left and which is on the right

    if rightmost_vertex_P[0] < leftmost_vertex_Q[0]:

        left_polygon = poly_P
        right_polygon = poly_Q
        
        # Midpoint between the rightmost of the left and leftmost of the right

        x_distance = (leftmost_vertex_Q[0] - rightmost_vertex_P[0]) / 2
        x_coordinate = x_distance + rightmost_vertex_P[0]
        
    else:

        left_polygon = poly_Q
        right_polygon = poly_P
        
        # Midpoint between the rightmost of the left and leftmost of the right

        y_distance = (leftmost_vertex_P[0] - rightmost_vertex_Q[0]) / 2
        x_coordinate = y_distance + leftmost_vertex_P[0]
    
    # Find the farthest top and bottom points on the y-axis for both polygons

    max_y_left_polygon = max(p[1] for p in left_polygon)
    min_y_left_polygon = min(p[1] for p in left_polygon)

    max_y_right_polygon = max(p[1] for p in right_polygon)
    min_y_right_polygon = min(p[1] for p in right_polygon)

    topmost_point = max(max_y_left_polygon, max_y_right_polygon)
    bottommost_point = min(min_y_left_polygon, min_y_right_polygon)

    # Create a vertical line using the calculated coordinates
    # (the -1 is added in order to always include the max vertices)

    top_point = topmost_point + 1
    bottom_point = bottommost_point - 1

    vertical_line = [(x_coordinate, top_point), (x_coordinate, bottom_point)]

    return vertical_line


# STEP 2.2 : the vertices, transition and L1-projection points of the polygon  
#            visible by the horizontal line, meaning connected by 1 vertical line,
#            that doesn't enter the interior areas of the polygon

def find_l1_set(poly, transition_points, L1_projection_points, horizontal_line):
    
    l1_set = set()
    
    for vertex in poly:

        l1_set.add(vertex)

    for point in transition_points:

        l1_set.add(point)

    for projection_point in L1_projection_points:

        l1_set.add(projection_point)

    # Remove all occurrences of None from the l1_set in place
    l1_set.difference_update({None})

    # If the polygon is below the horizontal line, exclude vertices that have an edge above them, 
    # meaning a pair of consecutive vertices that have greater y-coordinate, and lesser-greater x-coordinate

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

    # If the polygon is above the horizontal line, exclude vertices that have an edge below them, 
    # meaning a pair of consecutive vertices that have lesser y-coordinate, and lesser-greater x-coordinate

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


#            the vertices, transition and L1-projection points of the polygon  
#            visible by the vertical line, meaning connected by 1 horizontal line,
#            that doesn't enter the interior areas of the polygon

def find_l2_set(poly, transition_points, L1_projection_points, vertical_line):
    
    l2_set = set()

    for vertex in poly:

        l2_set.add(vertex)

    for point in transition_points:

        l2_set.add(point)

    for projection_point in L1_projection_points:

        l2_set.add(projection_point)

    # Remove all occurrences of None from the l2_set in place
    l2_set.difference_update({None})

    # If the polygon is right of the vertical line, exclude vertices that have an edge left of them, 
    # meaning a pair of consecutive vertices that have lesser x-coordinate, and lesser-greater y-coordinate

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

    # If the polygon is left of the vertical line, exclude vertices that have an edge right of them, 
    # meaning a pair of consecutive vertices that have greater y-coordinate, and lesser-greater x-coordinate

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


# STEP 3

#   m(x) = min[(x belongs in I1/2) l(P/Q)] m(x)
#   m(p) = L1(p,f(p)) + |xp| + |yp| 
#   L1 the distance of the path between p and its farthest neighbor f(p)
#   |xp|,|yp| the distance on the x-axis and y-axis respectively 
#   We find the point with the minimum m price of the set

def generate_candidates(point_set, neighbor_set, horizontal_line, vertical_line):

    # the variants that hold the point with the mu price and its corresponding point 

    min_price = float('inf') 
    min_price_point = None 

    # we check all the points of the given set, to find every mu, in order to choose the min

    for point in point_set:

        # first find the distance from the dividing lines

        distance_horizontal = abs(point[1] - horizontal_line[0][1])
        distance_vertical = abs(point[0] - vertical_line[0][0])

        max_path = find_most_distant_neighbor_path(point, neighbor_set)

        # add the 3 values you found

        total_distance = distance_horizontal + distance_vertical + max_path

        # compare to the current min price, and replace it if the new is lesser

        if (total_distance < min_price):

            min_price = total_distance
            min_price_point = point

        # return poth the point and the price

    return min_price_point, min_price


def find_most_distant_neighbor_path(point, neighbor_set):

# find the 4 most distant vertices of the polygon from the neighbor_set

    # for the left

    sorted_left_vertices = sorted(neighbor_set, key=lambda point: point[0])
    left_vertices = sorted_left_vertices[:2]
    
    sorted_left_vertices_by_y = sorted(left_vertices, key=lambda point: point[1])

    left_below = sorted_left_vertices_by_y[0] 
    left_up = sorted_left_vertices_by_y[1]

    # for the right

    sorted_right_vertices = sorted(neighbor_set, key=lambda point: point[0], reverse=True)
    right_vertices = sorted_right_vertices[:2]

    sorted_right_vertices_by_y = sorted(right_vertices, key=lambda point: point[1])

    right_below = sorted_right_vertices_by_y[0] 
    right_up = sorted_right_vertices_by_y[1]

# set the variants

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

    # compare the chosen point to every couple of corresponding vertices of the polygon
    # count the vertices and the total distanace you pass along the edges,
    # until you find the vertices that define the edge that contains the chosen point 

    for v1, v2 in paired_vertices:

        countVertices += 1

        # when you have not found the vertices that define the edge 
        # that contains the chosen point, check if you are on it

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
                        
        # when you already found the vertices that define the edge that contains the chosen point,
        # when you find a most distand vertice, since one of those 4 is the farthest neighbor,
        # if it is the first you find, save it, in order to know which vertices you missed

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

        # if you have not seen all the vertices, start again 
        # until you have counted the distances to each of them 
        
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
    
                # for safety an additional check

            if (countDistant == 4):

                break;

    # now count the actual distance, of all the edges of the polygon

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

    # the distanceTo_ is the path from left to right
    # the second_distanceTo_ is the path from right to left        

    second_distanceToFirst = total_edge_length - distanceToFirst
    second_distanceToSecond = total_edge_length - distanceToSecond
    second_distanceToThird = total_edge_length - distanceToThird
    second_distanceToFourth = total_edge_length - distanceToFourth

    # choose the max path, of the 2 

    max_distanceToFirst = max(distanceToFirst, second_distanceToFirst)
    max_distanceToSecond = max(distanceToSecond, second_distanceToSecond)
    max_distanceToThird = max(distanceToThird, second_distanceToThird)
    max_distanceToFourth = max(distanceToFourth, second_distanceToFourth)
        
    # choose the min path, of the 4, since it is the farthest neighbor
        
    max_path = min(max_distanceToFirst, max_distanceToSecond, max_distanceToThird, max_distanceToFourth)
    
    return max_path


# STEP 4

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
    
    '''
    print("F2_1 \n", F2_1, "\n")
    print("F2_2 \n", F2_2, "\n")
    '''

    # If [F2(p1,q2) less than or equal to F2 (p2,q1)] : return euclidean(p1,r) + euclidean(r,p2), where r=(xp1,yq2)
    # else : return euclidean(p2,r) + euclidean(r,q1), where r=(xq1,yq2)

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


# the euclidean distance is shown on the variables as an upper line

def euclidean_distance(a, b):
    
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
    


# ALG 1

def find_add_connectable_points(poly_1, poly_2, transition_points, L1_projection_points):
    
    connectable_points = set()

    horizontal_points_poly_1 = []
    vertical_points_poly_1 = []

    # Find pairs of consecutive vertices in poly_1 forming horizontal lines

    poly_1_list = list(poly_1)
    paired_points = zip(poly_1_list, poly_1_list[1:])

    horizontal_lines_poly_1 = {(p1, p2) for p1, p2 in paired_points if (p1[1] == p2[1])}
    
    # Find pairs of consecutive vertices in poly_1 forming vertical lines

    poly_1_list = list(poly_1)
    paired_points_vertical = zip(poly_1_list, poly_1_list[1:])

    vertical_lines_poly_1 = {(p1, p2) for p1, p2 in paired_points_vertical if (p1[0] == p2[0])}
    
    # Add the first and last points of poly_1 to horizontal_lines_poly_1

    poly_1_list = list(poly_1)
    first_point_1 = poly_1_list[0]

    poly_1_list = list(poly_1)
    last_point_1 = poly_1_list[-1]

    # Add the tuples to horizontal_lines_poly_1

    vertical_lines_poly_1.add((last_point_1, first_point_1))

    # Find pairs of consecutive vertices in poly_2 forming horizontal lines

    paired_points_horizontal = zip(poly_2, poly_2[1:])
    horizontal_lines_poly_2 = {(p1, p2) for p1, p2 in paired_points_horizontal if (p1[1] == p2[1])}
    
    # Find pairs of consecutive vertices in poly_2 forming vertical lines

    paired_points_vertical = zip(poly_2, poly_2[1:])
    vertical_lines_poly_2 = {(p1, p2) for p1, p2 in paired_points_vertical if (p1[0] == p2[0])}    
    
    # Add the first and last points of poly_1 to horizontal_lines_poly_1

    first_point_2 = list(poly_2)[0]
    last_point_2 = list(poly_2)[-1]

    # Add the tuples to horizontal_lines_poly_1

    vertical_lines_poly_2.add((last_point_2, first_point_2))
    
    '''
    print(horizontal_lines_poly_1)   
    print(vertical_lines_poly_1)
    '''
    
    for point in horizontal_lines_poly_1:
        
        horizontal_points_poly_1.append(point[0])
        horizontal_points_poly_1.append(point[1])
    
    for point in vertical_lines_poly_1:
        
        vertical_points_poly_1.append(point[0])
        vertical_points_poly_1.append(point[1])
    
    # Check which set transition_points belong to and add points to the respective list
    
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

    # Check which set L1_projection_points belong to and add points to the respective list
    
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
    
    #print(horizontal_points_poly_1)   
    #print(vertical_points_poly_1,"\n")
    
    # Iterate through each point in horizontal_points_poly_1

    for point1 in horizontal_points_poly_1:
        
        x1 = point1[0]
        y1 = point1[1]

        check = 0

        # Check if there exists a pair in horizontal_lines_poly_2
        
        matching_pairs = {(p1, p2) for (p1, p2) in horizontal_lines_poly_2 if ((p1[0] <= x1 <= p2[0]) or (p1[0] >= x1 >= p2[0]))}

        # Check conditions for matching_point
        
        for p1, p2 in matching_pairs:
            
            check = 0
            
            matching_point = (x1, p1[1])

            x_matching = matching_point[0]
            y_matching = matching_point[1]

            # Check conditions for horizontal_lines_poly_2

            for (x_p1, y_p1), (x_p2, y_p2) in horizontal_lines_poly_2:
                
                condition_1 = (x_p1 < x_matching < x_p2) or (x_p1 > x_matching > x_p2) or (x_p1 == x_matching) or (x_matching == x_p2) 
                condition_2 = (y1 < y_p1 < y_matching) or (y1 > y_p1 > y_matching)

                if (condition_1) and (condition_2):
                    
                    check=1

            # Check conditions for horizontal_lines_poly_1

            for (x_p1, y_p1), (x_p2, y_p2) in horizontal_lines_poly_1:
                
                condition_1 = (x_p1 < x_matching < x_p2) or (x_p1 > x_matching > x_p2) or (x_p1 == x_matching) or (x_matching == x_p2)
                condition_2 = (y1 < y_p1 < y_matching) or (y1 > y_p1 > y_matching)

                if (condition_1) and (condition_2):
                
                    check=1
                    
            if (check==0):

                connectable_points.add((point1, matching_point))

        #print("h cp: ",connectable_points,"\n")

    # Iterate through each pair in vertical_lines_poly_1

    for point1 in vertical_points_poly_1:
        
        x1 = point1[0]
        y1 = point1[1]

        check = 0
    
        # Check if there exists a pair in vertical_lines_poly_2

        matching_pairs = {(p1, p2) for (p1, p2) in vertical_lines_poly_2 if (p1[1] <= y1 <= p2[1]) or (p1[1] >= y1 >= p2[1])}
    
        # Check conditions for matching_point

        for p1, p2 in matching_pairs:
                
            matching_point = (p1[0], y1)

            x_matching = matching_point[0]
            y_matching = matching_point[1]
    
            # Check conditions for vertical_lines_poly_2

            for (x_p1, y_p1), (x_p2, y_p2) in vertical_lines_poly_2:
                    
                condition_1 = (y_p1 < y_matching < y_p2) or (y_p1 > y_matching > y_p2) or (y_p1 == y_matching) or (y_matching == y_p2)
                condition_2 = (x1 < x_p1 < x_matching) or (x1 > x_p1 > x_matching)
    
                if (condition_1) and (condition_2):    
                    
                    check=1
    
            # Check conditions for vertical_lines_poly_1

            for (x_p1, y_p1), (x_p2, y_p2) in vertical_lines_poly_1:
                    
                condition_1 = (y_p1 < y_matching < y_p2) or (y_p1 > y_matching > y_p2) or (y_p1 == y_matching) or (y_matching == y_p2)
                condition_2 = (x1 < x_p1 < x_matching) or (x1 > x_p1 > x_matching)
    
                if (condition_1) and (condition_2):                
                    
                    check=1
                    
            if (check == 0):

                connectable_points.add((point1, matching_point))

        #print("v cp: ",connectable_points,"\n")

    return connectable_points


def find_F1(points_pairs_set, neighbor_set):

    point1 = points_pairs_set[0]
    point2 = points_pairs_set[1]

    # the variants that hold the point with the mu price and its corresponding point 

    F1_list_prices = set()

    # first find the distance of the 2 points

    distance_horizontal = abs(point1[0] - point2[0])
    distance_vertical = abs(point1[1] - point2[1])

    max_path = find_most_distant_neighbor_path(point1, neighbor_set)

    # add the 3 values you found

    total_distance = distance_horizontal + distance_vertical + max_path

    # add to the list

    F1_list_prices.add(total_distance)

    # return the price, since we have the points on a list

    return F1_list_prices


def remove_duplicates(Bridge_Alg1):

    # Convert Bridge_Alg1 back to a list (removing duplicates)
    
    unique_bridge_set = sort_list(Bridge_Alg1)

    Bridge_Alg1 = sort_list(unique_bridge_set)

    return Bridge_Alg1


def sort_list(list):

     # Create an empty list to store the sorted pairs

    sorted_pairs = []

    # Iterate through each pair in Bridge_Alg1 and append the sorted pair to the list

    for pair in list:

        sorted_pair = tuple(sorted(pair))

        sorted_pairs.append(sorted_pair)

    # Create a set from the list to remove duplicates
        
    unique_list_set = set(sorted_pairs)

    return unique_list_set


def find_F1_for_list(poly, connectable_points):

    # Create an empty list to store the results

    F1_list = []

    # Iterate through each pair in connectable_points_P and append the result to the list

    for pair in connectable_points:

        result = find_F1(pair, poly)

        F1_list.append(result)

    return F1_list


def F1_min(F1_list_P, F1_list_Q, connectable_points_P, connectable_points_Q):

    # Find the minimum price and its corresponding pair

    zipped_values = zip(F1_list_P, connectable_points_P)
    min_value_pair = min(zipped_values, key=lambda x: x[0])

    F1_min_P_price = min_value_pair[0]
    F1_min_P_point = min_value_pair[1]

    zipped_values_Q = zip(F1_list_Q, connectable_points_Q)
    min_value_pair_Q = min(zipped_values_Q, key=lambda x: x[0])

    F1_min_Q_price = min_value_pair_Q[0]
    F1_min_Q_point = min_value_pair_Q[1]

    # Compare the minimum prices and select the overall minimum

    if (F1_min_P_price < F1_min_Q_price):

        F1_min_price = F1_min_P_price
        F1_min_point = F1_min_P_point

    else:

        F1_min_price = F1_min_Q_price
        F1_min_point = F1_min_Q_point

    # Extract the bridge price from the result
    F1_min_price = F1_min_price.pop()

    result = (F1_min_price, F1_min_point[0], F1_min_point[1])
    
    return result



# Example usage rectilinear_polygon:


# CASE 1:

poly_P = [(0,5.5), (1,5.5), (1,2), (2,2), (2,3), (3,3), (3,2), (4,2), (4,1), (0,1)]

poly_Q = [(3,6), (4,6), (4,7), (6,7), (6,8), (7,8), (7,5), (6,5), (6,4), (4,4), (4,5), (3,5)]

find_Bridge(poly_P, poly_Q)

# CASE 2: 

poly_P = [(0, 0),(0, 5),(2, 5),(2, 3),(6, 3),(6, 4),(7, 4),(7, 2),(8, 2),(8, 0)]

poly_Q = [(10, 7), (10, 8), (12, 8), (12, 10), (14, 10), (14, 8), (15, 8), (15, 7)]

find_Bridge(poly_P, poly_Q)