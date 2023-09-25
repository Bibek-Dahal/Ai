from django.shortcuts import redirect, render
from collections import deque

# from collections import deque




def render_state_space_tree(request):
    class PuzzleState:
        def __init__(self, state, parent=None, move=""):
            self.state = state
            self.parent = parent
            # self.move = move

        def __eq__(self, other):
            return self.state == other.state

    def generate_successors(state):
        successors = []
        zero_index = state.state.index(0) #return index of list where 0 is present
        moves = [-1, 1, -3, 3]  # Left, Right, Up, Down

        for move in moves:
            new_index = zero_index + move

            if (
                0 <= new_index < len(state.state)
                and not (
                    (zero_index % 3 == 0 and move == -1)  # Check if moving left is valid
                    or (zero_index % 3 == 2 and move == 1)  # Check if moving right is valid
                    )
            ):
                new_state = state.state[:] #makes a shallow copy
                new_state[zero_index], new_state[new_index] = new_state[new_index], new_state[zero_index]
                successors.append(PuzzleState(new_state, state))
        
        return successors

    def bfs(initial_state, goal_state):
        queue = deque([(initial_state, 0)])  # Include depth level
        visited = set([tuple(initial_state.state)])
        all_states = {}  # To store all explored states

        while queue:
            current_state, depth = queue.popleft()

            if depth not in all_states:
                all_states[depth] = []

            all_states[depth].append(current_state)  # Store the state object itself at the depth

            if current_state.state == goal_state.state:
                return all_states
            
            successors = generate_successors(current_state)
            for successor in successors:
                if tuple(successor.state) not in visited:
                    visited.add(tuple(successor.state))
                    queue.append((successor, depth + 1))  # Include depth level

    
    if request.method == 'GET':
        initial = [2, 8, 3, 1, 6, 4, 7, 0, 5]
        goal = [2, 0, 8, 1, 6, 3, 7, 5, 4]
        initial_state = PuzzleState(initial)
        goal_state = PuzzleState(goal)

        explored_states = bfs(initial_state, goal_state)

        # Prepare the data for vis.js network
        nodes = []
        edges = []

        for depth, states_at_depth in explored_states.items():
            for state in states_at_depth:
                # Add nodes
                nodes.append({"id": str(state.state), 'label':str(state.state),"level": f"{depth}"})
                
                # nodes.append({"id": str(state.state), 'label':str(state.state)})

                # Add edges (parent-child relationship)
                if state.parent:
                    edges.append({"from": str(state.parent.state), "to": str(state.state), 'arrows': 'to'})

        context = {"nodes": nodes, "edges": edges}
        return render(request, "index.html", context)
    if request.method == "POST":
        print("initial",[ int(x) for x in request.POST.get('initial').split(' ')])
        initial = [int(x) for x in request.POST.get('initial').split(' ')]
        goal = [int(x) for x in request.POST.get('final').split(' ')]
        initial_state = PuzzleState(initial)
        goal_state = PuzzleState(goal)

        explored_states = bfs(initial_state, goal_state)

        # Prepare the data for vis.js network
        nodes = []
        edges = []

        for depth, states_at_depth in explored_states.items():
            for state in states_at_depth:
                # Add nodes
                nodes.append({"id": str(state.state), 'label':str(state.state),"level": f"{depth}"})
                
                # nodes.append({"id": str(state.state), 'label':str(state.state)})

                # Add edges (parent-child relationship)
                if state.parent:
                    edges.append({"from": str(state.parent.state), "to": str(state.state), 'arrows': 'to'})

        context = {"nodes": nodes, "edges": edges}
        return render(request, "index.html", context)


def renderDfs(request):
    class PuzzleState:
        def __init__(self, state, parent=None, move=""):
            self.state = state
            self.parent = parent
            self.move = move

        def __eq__(self, other):
            return self.state == other.state

    def generate_successors(state):
        successors = []
        zero_index = state.state.index(0)  # Return index of list where 0 is present
        moves = [-1, 1, -3, 3]  # Left, Right, Up, Down

        for move in moves:
            new_index = zero_index + move

            if (
                0 <= new_index < len(state.state)
                and not (
                    (zero_index % 3 == 0 and move == -1)  # Check if moving left is valid
                    or (zero_index % 3 == 2 and move == 1)  # Check if moving right is valid
                )
            ):
                new_state = state.state[:]  # Makes a shallow copy
                new_state[zero_index], new_state[new_index] = new_state[new_index], new_state[zero_index]
                successors.append(PuzzleState(new_state, state, move))  # Include the move

        return successors

    def dfs(initial_state, goal_state):
        stack = [(initial_state, 0)]  # Include depth level
        visited = set([tuple(initial_state.state)])
        all_states = {}  # To store all explored states

        while stack:
            current_state, depth = stack.pop()

            if depth not in all_states:
                all_states[depth] = []

            all_states[depth].append(current_state)  # Store the state object itself at the depth

            if current_state.state == goal_state.state:
                return all_states

            successors = generate_successors(current_state)
            if depth < 8:  # Limit the depth
                successors = generate_successors(current_state)
                for successor in successors:
                    if tuple(successor.state) not in visited:
                        visited.add(tuple(successor.state))
                        stack.append((successor, depth + 1))
    
    if request.method == 'GET':
        initial = [2, 8, 3, 1, 6, 4, 7, 0, 5]
        goal = [2, 0, 8, 1, 6, 3, 7, 5, 4]
        initial_state = PuzzleState(initial)
        goal_state = PuzzleState(goal)

        explored_states = dfs(initial_state, goal_state)
        print(len(explored_states))

        # Prepare the data for vis.js network
        nodes = []
        edges = []

        for depth, states_at_depth in explored_states.items():
            for state in states_at_depth:
                # Add nodes
                nodes.append({"id": str(state.state), 'label':str(state.state),"level": f"{depth}"})
                
                # nodes.append({"id": str(state.state), 'label':str(state.state)})

                # Add edges (parent-child relationship)
                if state.parent:
                    edges.append({"from": str(state.parent.state), "to": str(state.state), 'arrows': 'to'})

        context = {"nodes": nodes, "edges": edges}
        return render(request, "index.html", context)
    if request.method == "POST":
        print("initial",[ int(x) for x in request.POST.get('initial').split(' ')])
        initial = [int(x) for x in request.POST.get('initial').split(' ')]
        goal = [int(x) for x in request.POST.get('final').split(' ')]
        initial_state = PuzzleState(initial)
        goal_state = PuzzleState(goal)

        explored_states = dfs(initial_state, goal_state)
        print(len(explored_states))
        # Prepare the data for vis.js network
        nodes = []
        edges = []

        for depth, states_at_depth in explored_states.items():
            for state in states_at_depth:
                # Add nodes
                nodes.append({"id": str(state.state), 'label':str(state.state),"level": f"{depth}"})
                
                # nodes.append({"id": str(state.state), 'label':str(state.state)})

                # Add edges (parent-child relationship)
                if state.parent:
                    edges.append({"from": str(state.parent.state), "to": str(state.state), 'arrows': 'to'})

        context = {"nodes": nodes, "edges": edges}
        return render(request, "index.html", context)
    


def manhatten(request):
    import heapq

    # Define the goal state
    goal_state = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    class PuzzleState:
        def __init__(self, state, parent=None, move=""):
            self.state = state
            self.parent = parent
            self.move = move
            self.g = 0  # Cost to reach this node
            self.h = 0  # Estimated cost to reach the goal (Manhattan distance)
            self.f = 0  # Total cost (f = g + h)

            if parent:
                self.g = parent.g + 1
            self.h = self.calculate_manhattan_distance(goal_state)  # Pass goal_state as a parameter
            self.f = self.g + self.h

        def __eq__(self, other):
            return self.state == other.state

        def __lt__(self, other):
            return self.f < other.f

        def calculate_manhattan_distance(self, goal_state):
            # Calculate the Manhattan distance heuristic
            total_distance = 0
            for i in range(len(self.state)):
                if self.state[i] != 0:
                    current_row, current_col = i // 3, i % 3
                    goal_index = goal_state.index(self.state[i])
                    goal_row, goal_col = goal_index // 3, goal_index % 3
                    total_distance += abs(current_row - goal_row) + abs(current_col - goal_col)
            return total_distance

    def generate_successors(state):
        successors = []
        zero_index = state.state.index(0)
        moves = [-1, 1, -3, 3]  # Left, Right, Up, Down

        for move in moves:
            new_index = zero_index + move

            if (
                0 <= new_index < len(state.state)
                and not (
                    (zero_index % 3 == 0 and move == -1)  # Check if moving left is valid
                    or (zero_index % 3 == 2 and move == 1)  # Check if moving right is valid
                )
            ):
                new_state = state.state[:]
                new_state[zero_index], new_state[new_index] = new_state[new_index], new_state[zero_index]
                successor = PuzzleState(new_state, state, move)
                successors.append(successor)

        return successors

    def a_star(initial_state, goal_state):
        open_set = []
        closed_set = set()
        open_set.append(initial_state)

        explored_states = {}  # Dictionary to store explored states at each depth

        while open_set:
            current_node = min(open_set, key=lambda x: x.f)
            open_set.remove(current_node)
            closed_set.add(tuple(current_node.state))
            
            depth = current_node.g

            if depth not in explored_states:
                explored_states[depth] = []

            explored_states[depth].append(current_node.state)

            if current_node.state == goal_state:
                return explored_states  # Goal node found, return the explored states

            successors = generate_successors(current_node)
            for successor in successors:
                if tuple(successor.state) in closed_set:
                    continue

                if successor not in open_set:
                    open_set.append(successor)

        return None  # If the loop completes without finding the goal node, return None


    # Define the initial state
    initial_state = PuzzleState([7,2,4,5,0,6,8,3,1])

    # Perform A* search
    result = a_star(initial_state, goal_state)

    # Prepare the data for vis.js network
    nodes = []
    edges = []

    if result:
        for depth, states_at_depth in result.items():
            for state in states_at_depth:
                # Add nodes
                    nodes.append({"id": str(state.state), 'label':str(state.state),"level": f"{depth}"})
                    
                    # nodes.append({"id": str(state.state), 'label':str(state.state)})

                    # Add edges (parent-child relationship)
                    if state.parent:
                        edges.append({"from": str(state.parent.state), "to": str(state.state), 'arrows': 'to'})

        context = {"nodes": nodes, "edges": edges}
        return render(request, "index.html", context)
        print(context)
    else:
        print("No solution found.")


