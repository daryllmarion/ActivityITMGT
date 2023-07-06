'''Individual Programming Assignment 3

70 points

This assignment will develop your ability to manipulate data.
'''

def relationship_status(from_member, to_member, social_graph):
    '''Relationship Status.
    20 points.

    Let us pretend that you are building a new app.
    Your app supports social media functionality, which means that users can have
    relationships with other users.

    There are two guidelines for describing relationships on this social media app:
    1. Any user can follow any other user.
    2. If two users follow each other, they are considered friends.

    This function describes the relationship that two users have with each other.

    Please see "assignment-4-sample-data.py" for sample data. The social graph
    will adhere to the same pattern.

    Parameters
    ----------
    from_member: str
        the subject member
    to_member: str
        the object member
    social_graph: dict
        the relationship data

    Returns
    -------
    str
        "follower" if fromMember follows toMember,
        "followed by" if fromMember is followed by toMember,
        "friends" if fromMember and toMember follow each other,
        "no relationship" if neither fromMember nor toMember follow each other.
    '''
    # Replace `pass` with your code.
    # Stay within the function. Only use the parameters as input. The function should return your answer.
    status = ""
    if to_member in social_graph.get(from_member).get("following"):
        if from_member in social_graph.get(to_member).get("following"):
            status = "friends"
        else:
            status = "follower"
    elif from_member in social_graph.get(to_member).get("following"):
        status = "followed by"
    else:
        status = "no relationship"
    return status


def tic_tac_toe(board):
    '''Tic Tac Toe.
    25 points.

    Tic Tac Toe is a common paper-and-pencil game.
    Players must attempt to successfully draw a straight line of their symbol across a grid.
    The player that does this first is considered the winner.

    This function evaluates a tic tac toe board and returns the winner.

    Please see "assignment-4-sample-data.py" for sample data. The board will adhere
    to the same pattern. The board may by 3x3, 4x4, 5x5, or 6x6. The board will never
    have more than one winner. The board will only ever have 2 unique symbols at the same time.

    Parameters
    ----------
    board: list
        the representation of the tic-tac-toe board as a square list of lists

    Returns
    -------
    str
        the symbol of the winner or "NO WINNER" if there is no winner
    '''
    # Replace `pass` with your code.
    # Stay within the function. Only use the parameters as input. The function should return your answer.
    size = len(board)
    checklist = []
    win = False 
    
    #Checking Horizontal...
    for row in range(size):
        checkset = set(board[row])
        if len(checkset) == 1 and '' not in checkset:
            win = True
            return checkset.pop()
        else:
            win = False
    
        
    #Checking Vertical
    for row in range(size):
        for column in range(size):
            checklist.append(board[column][row])
        
        checkset = set(checklist)
        checklist = []
        if len(checkset) == 1 and '' not in checkset:
            win = True 
            return checkset.pop()
        else:
            win = False

    #Checking Left Diagonal
    
    for leftdiag in range(size):
        checklist.append(board[leftdiag][leftdiag])
        
    checkset = set(checklist)
    if len(checkset) == 1 and '' not in checklist:
        win = True
        return checkset.pop() 
    
    checklist = []
    
    
    #Checking Right Diagonal
    
    rightdiag = size - 1
    col = 0

    while rightdiag != -1:
        checklist.append(board[col][rightdiag])
        col += 1
        rightdiag -= 1
         
    checkset = set(checklist)
    if len(checkset) == 1 and '' not in checklist:
        win = True 
        return checkset.pop()
    
    checklist = []
        
    
    #Final result output
   
    if win == False:
        return("NO WINNER")

def eta(first_stop, second_stop, route_map):
    '''ETA.
    25 points.

    A shuttle van service is tasked to travel along a predefined circlar route.
    This route is divided into several legs between stops.
    The route is one-way only, and it is fully connected to itself.

    This function returns how long it will take the shuttle to arrive at a stop
    after leaving another stop.

    Please see "mod-4-ipa-1-sample-data.py" for sample data. The route map will
    adhere to the same pattern. The route map may contain more legs and more stops,
    but it will always be one-way and fully enclosed.

    Parameters
    ----------
    first_stop: str
        the stop that the shuttle will leave
    second_stop: str
        the stop that the shuttle will arrive at
    route_map: dict
        the data describing the routes

    Returns
    -------
    int
        the time it will take the shuttle to travel from first_stop to second_stop
    '''
    # Replace `pass` with your code.
    # Stay within the function. Only use the parameters as input. The function should return your answer.
    total_travel_time = 0
    
    if first_stop == second_stop: #if round trip
        total_travel_time = 0
        return total_travel_time
        
    if (first_stop, second_stop) in route_map: #one-stop
        return route_map[(first_stop, second_stop)]['travel_time_mins']    
        
    # Iterate over all possible routes
    for key in route_map.keys():
        if key[0] == first_stop:
            # Recursively check if there is a route from the intermediate stop to the second stop
            total_travel_time = eta(key[1], second_stop, route_map)
            
            # If a route is found, return the total travel time
            if total_travel_time is not None:
                return route_map[key]['travel_time_mins'] + total_travel_time
