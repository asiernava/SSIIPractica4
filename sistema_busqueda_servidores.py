"""
================================================================================
SISTEMA INTELIGENTE DE BÚSQUEDA DE RUTAS EN CENTRO DE DATOS
================================================================================

PROBLEMA:
Un grupo de servidores ha estado teniendo problemas ya que el envío de datos 
entre servidores era altamente ineficiente, lo que causaba retrasos importantes 
y causaba una mala experiencia para los clientes.

CONTEXTO Y FINALIDAD:
El sistema de búsquedas desarrollado pretende agilizar el intercambio de 
información entre servidores buscando las rutas más óptimas entre los mismos 
mediante distintos algoritmos.

REPRESENTACIÓN DEL PROBLEMA:
Cada posición dentro de la matriz representa un servidor físico dentro del 
centro de datos. El objetivo es trasladar la carga de trabajo crítica desde 
el servidor de origen (START) hasta el servidor destino (END), utilizando un 
camino válido y eficiente. Algunos servidores presentan fallos, mantenimiento 
o sobrecarga extrema, y por ello se representan como 'X', indicando que no 
pueden ser utilizados como ruta.

La matriz tiene dos niveles conectados por portales 'UP' y 'DOWN' que permiten
el movimiento entre niveles.

ALGORITMOS IMPLEMENTADOS:
- BFS (Breadth-First Search) - Tree y Graph
- DFS (Depth-First Search) - Tree y Graph
- Greedy (Best-First Search) - Tree y Graph
- A* (A-Star) - Tree y Graph

================================================================================
"""

# ==============================================================================
# SECCIÓN 1: INTRODUCCIÓN Y ESTRUCTURA DE DATOS
# ==============================================================================

# Matriz del centro de datos (2 niveles de 10x10)
matriz = [
    [   
        ['14', 'X',  '07', '88', '03', 'X',  '45', '12', 'X',  '30'],
        ['91', '55', '10', 'X',  '72', '61', '19', '84', '05', '22'],
        ['X',  'X',  '48', '31', '97', '06', 'X',  '73', '29', '13'],
        ['80', '52', 'X',  '94', '21', 'X',  '67', '11', 'X',  '32'],
        ['08', '76', 'X',  '44', '57', 'UP',  '90', '23', '04', '63'],
        ['X',  '26', '59', 'X',  '70', '41', '18', '39', '15', 'X'],
        ['87', '53', '24', 'X',  '92', '60', '78', 'X',  '36', '17'],
        ['X',  '02', '95', '50', 'X',  '66', '27', '46', 'X',  '58'],
        ['09', '43', '85', '07', 'X',  'X',  '34', '79', '25', 'X'],
        ['16', 'X',  '64', '38', '99', '12', '88', '55', '03', 'X']
    ],
    [   
        ['134','107','188','103','X',  '145','112','X',  '130','191'],
        ['X',  '155','110','X',  '172','161','119','184','105','122'],
        ['X',  '148','131','197','106','X',  '173','129','113','180'],
        ['152','X',  '194','121','X',  '167','111','X',  '132','108'],
        ['176','X',  '144','157','DOWN',  '190','123','104','X',  '163'],
        ['X',  '126','159','X',  '170','141','118','139','115','X'],
        ['187','153','124','X',  '192','160','178','X',  '136','117'],
        ['X',  '102','195','150','X',  '166','127','146','X',  'X'],
        ['158','109','143','185','137','X',  '133','179','125','X'],
        ['116','X',  '164','138','199','101','102','103','104','105']
    ]
]

# Diccionarios de orígenes y destinos para pruebas
Lista_Origenes = {
    '55': (0, 1, 2),
    '30': (0, 0, 9),
    '07': (0, 0, 2),
    '145': (1, 0, 5),
    '194': (1, 3, 2),
    '26': (0, 5, 1),
    '78': (0, 6, 6)
}

Lista_Destinos = {
    '43': (0, 8, 1),
    '58': (0, 7, 9),
    '60': (0, 6, 5),
    '150': (1, 7, 3),
    '116': (1, 9, 0),
    '159': (1, 5, 2),
    '148': (1, 2, 1)
}

# ==============================================================================
# SECCIÓN 2: CÓDIGO GENÉRICO - CLASES Y FUNCIONES COMPARTIDAS
# ==============================================================================

class Node:
    """Clase que representa un nodo en el árbol/grafo de búsqueda"""
    
    def __init__(self, parent=None, state=None, cost=0, action=None, depth=0, heuristic=0):
        self.parent = parent
        self.state = state
        self.cost = cost
        self.action = action
        self.depth = depth
        self.heuristic = heuristic
    
    def __str__(self):
        return str(self.state)
    
    def set_parent(self, parent):
        self.parent = parent
    
    def get_parent(self):
        return self.parent
    
    def set_state(self, state):
        self.state = state
    
    def get_state(self):
        return self.state
    
    def set_cost(self, cost):
        self.cost = cost
    
    def get_cost(self):
        return self.cost
    
    def set_action(self, action):
        self.action = action
    
    def get_action(self):
        return self.action
    
    def set_depth(self, depth):
        self.depth = depth
    
    def get_depth(self):
        return self.depth
    
    def set_heuristic(self, heuristic):
        self.heuristic = heuristic
    
    def get_heuristic(self):
        return self.heuristic
    
    def get_f_score(self):
        """Retorna f(n) = g(n) + h(n) para A*"""
        return self.cost + self.heuristic


def get_neighbors(matriz, position, goal_number):
    """
    Devuelve las posiciones vecinas válidas desde una posición dada.
    Maneja el cambio de nivel a través de portales UP/DOWN.
    
    ORDEN DE VISITA EXPLÍCITO (para algoritmos no informados):
    Los vecinos se generan en el siguiente orden:
    1. Derecha (0, +1)
    2. Abajo (+1, 0)
    3. Izquierda (0, -1)
    4. Arriba (-1, 0)
    5. Cambio de nivel (UP/DOWN) si corresponde
    """
    nivel, row, col = position
    rows = len(matriz[nivel])
    cols = len(matriz[nivel][0])
    neighbors = []
    
    goal_number = int(goal_number)
    target_level = 1 if goal_number > 99 else 0
    
    # Si estamos en el nivel equivocado, buscar el portal
    # ORDEN: Derecha, Abajo, Izquierda, Arriba
    if nivel != target_level:
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for dr, dc in directions:
            new_row = row + dr
            new_col = col + dc
            if 0 <= new_row < rows and 0 <= new_col < cols:
                if matriz[nivel][new_row][new_col] != 'X':
                    neighbors.append((nivel, new_row, new_col))
        
        current_value = matriz[nivel][row][col]
        
        if current_value == 'UP' and nivel + 1 < len(matriz):
            neighbors.append((nivel + 1, row, col))
        
        if current_value == 'DOWN' and nivel - 1 >= 0:
            neighbors.append((nivel - 1, row, col))
        
        return neighbors
    
    # Si estamos en el nivel correcto, moverse normalmente
    # ORDEN: Derecha, Abajo, Izquierda, Arriba
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    for dr, dc in directions:
        new_row = row + dr
        new_col = col + dc
        if 0 <= new_row < rows and 0 <= new_col < cols:
            if matriz[nivel][new_row][new_col] != 'X':
                neighbors.append((nivel, new_row, new_col))
    
    return neighbors


def expand(node_parent, matriz, goal_number):
    """
    Expande un nodo generando sus sucesores.
    Devuelve una lista de nodos hijo.
    """
    children = []
    for neighbor in get_neighbors(matriz, node_parent.get_state(), goal_number):
        child = Node()
        child.set_state(neighbor)
        child.set_parent(node_parent)
        child.set_action(get_neighbors(matriz, neighbor, goal_number))
        child.set_cost(node_parent.get_cost() + 1)
        child.set_depth(node_parent.get_depth() + 1)
        children.append(child)
    return children


def calculate_manhattan_distance(pos1, pos2):
    """Calcula la distancia Manhattan entre dos posiciones (nivel, fila, columna)"""
    nivel1, row1, col1 = pos1
    nivel2, row2, col2 = pos2
    
    # Penalización por cambio de nivel
    level_penalty = abs(nivel1 - nivel2) * 5
    
    return abs(row1 - row2) + abs(col1 - col2) + level_penalty


def reconstruct_path(node):
    """Reconstruye el camino desde el nodo inicial hasta el nodo objetivo"""
    path = []
    temp_node = node
    while temp_node:
        path.append(temp_node.get_state())
        temp_node = temp_node.get_parent()
    path.reverse()
    return path


def print_solution(found, node, algorithm_name):
    """Imprime la solución encontrada de forma estandarizada"""
    if found:
        path = reconstruct_path(node)
        print(f"\n{'='*60}")
        print(f"¡OBJETIVO ENCONTRADO CON {algorithm_name}!")
        print(f"{'='*60}")
        print(f"Camino: {path}")
        print(f"Coste total: {node.get_cost()}")
        print(f"Profundidad: {node.get_depth()}")
        print(f"Nodos en el camino: {len(path)}")
        print(f"{'='*60}\n")
        return node
    else:
        print(f"\n{algorithm_name}: Solución no encontrada\n")
        return None


# ==============================================================================
# SECCIÓN 3: BFS (BREADTH-FIRST SEARCH)
# ==============================================================================

def bfs_tree(matriz, start, goal_number):
    """
    BFS Tree Search: Búsqueda en anchura sin control de estados repetidos.
    Explora por niveles, garantiza camino más corto si todos los costes son iguales.
    
    ORDEN DE VISITA: FIFO (First In, First Out)
    - Se extraen nodos con pop(0) - el más antiguo primero
    - Los vecinos se añaden al final de la frontera
    - Resultado: Exploración por niveles de profundidad
    """
    frontier = []
    found = False
    iteration_limit = 1000
    iterations = 0
    
    node1 = Node()
    node1.set_parent(None)
    node1.set_state(start)
    node1.set_action(get_neighbors(matriz, start, goal_number))
    node1.set_cost(0)
    node1.set_depth(0)
    frontier.append(node1)
    
    print("\n--- INICIO BFS TREE SEARCH ---")
    
    while not found and frontier and iterations < iteration_limit:
        iterations += 1
        node_parent = frontier.pop(0)
        current_state = node_parent.get_state()
        
        print(f"\nIteración {iterations} - Expandiendo: {current_state}")
        print(f"  Coste: {node_parent.get_cost()}, Profundidad: {node_parent.get_depth()}")
        
        valor_actual = matriz[current_state[0]][current_state[1]][current_state[2]]
        if str(valor_actual) == str(goal_number):
            found = True
            node1 = node_parent
            break
        
        next_nodes = expand(node_parent, matriz, goal_number)
        frontier.extend(next_nodes)
        
        print(f"  Frontera: {len(frontier)} nodos")
    
    return print_solution(found, node1, "BFS TREE")


def bfs_graph(matriz, start, goal_number):
    """
    BFS Graph Search: Búsqueda en anchura con control de estados visitados.
    Evita ciclos y estados repetidos.
    
    ORDEN DE VISITA: FIFO (First In, First Out)
    - Se extraen nodos con pop(0) - el más antiguo primero
    - Los vecinos se añaden al final de la frontera
    - Solo se añaden estados no visitados previamente
    - Resultado: Exploración por niveles de profundidad sin repeticiones
    """
    goal_number = str(goal_number)
    frontier = []
    found = False
    iteration_limit = 1000
    iterations = 0
    explored = set()
    
    node1 = Node()
    node1.set_parent(None)
    node1.set_state(start)
    node1.set_action(get_neighbors(matriz, start, goal_number))
    node1.set_cost(0)
    node1.set_depth(0)
    frontier.append(node1)
    explored.add(start)
    
    print("\n--- INICIO BFS GRAPH SEARCH ---")
    
    while not found and frontier and iterations < iteration_limit:
        iterations += 1
        node_parent = frontier.pop(0)
        current_state = node_parent.get_state()
        
        print(f"\nIteración {iterations} - Expandiendo: {current_state}")
        print(f"  Coste: {node_parent.get_cost()}, Profundidad: {node_parent.get_depth()}")
        
        valor_actual = matriz[current_state[0]][current_state[1]][current_state[2]]
        if str(valor_actual) == goal_number:
            found = True
            node1 = node_parent
            break
        
        next_nodes = expand(node_parent, matriz, goal_number)
        
        for child in next_nodes:
            child_state = child.get_state()
            if child_state not in explored:
                explored.add(child_state)
                frontier.append(child)
        
        print(f"  Frontera: {len(frontier)} nodos, Explorados: {len(explored)}")
    
    return print_solution(found, node1, "BFS GRAPH")


# ==============================================================================
# SECCIÓN 4: DFS (DEPTH-FIRST SEARCH)
# ==============================================================================

def dfs_tree(matriz, start, goal_number):
    """
    DFS Tree Search: Búsqueda en profundidad sin control de estados repetidos.
    Explora hasta el fondo antes de retroceder.
    
    ORDEN DE VISITA: LIFO (Last In, First Out)
    - Se extraen nodos con pop() - el más reciente primero
    - Los vecinos se añaden al final de la frontera
    - Resultado: Exploración en profundidad (rama completa antes de cambiar)
    """
    frontier = []
    found = False
    iteration_limit = 1000
    iterations = 0
    
    node1 = Node()
    node1.set_parent(None)
    node1.set_state(start)
    node1.set_action(get_neighbors(matriz, start, goal_number))
    node1.set_cost(0)
    node1.set_depth(0)
    frontier.append(node1)
    
    print("\n--- INICIO DFS TREE SEARCH ---")
    
    while not found and frontier and iterations < iteration_limit:
        iterations += 1
        node_parent = frontier.pop()  # LIFO - Último en entrar, primero en salir
        current_state = node_parent.get_state()
        
        print(f"\nIteración {iterations} - Expandiendo: {current_state}")
        print(f"  Coste: {node_parent.get_cost()}, Profundidad: {node_parent.get_depth()}")
        
        valor_actual = matriz[current_state[0]][current_state[1]][current_state[2]]
        if str(valor_actual) == str(goal_number):
            found = True
            node1 = node_parent
            break
        
        next_nodes = expand(node_parent, matriz, goal_number)
        frontier.extend(next_nodes)
        
        print(f"  Frontera: {len(frontier)} nodos")
    
    return print_solution(found, node1, "DFS TREE")


def dfs_graph(matriz, start, goal_number):
    """
    DFS Graph Search: Búsqueda en profundidad con control de estados visitados.
    Evita ciclos y estados repetidos.
    
    ORDEN DE VISITA: LIFO (Last In, First Out)
    - Se extraen nodos con pop() - el más reciente primero
    - Los vecinos se añaden al final de la frontera
    - Solo se añaden estados no visitados previamente
    - Resultado: Exploración en profundidad sin repeticiones
    """
    goal_number = str(goal_number)
    frontier = []
    found = False
    iteration_limit = 1000
    iterations = 0
    explored = set()
    
    node1 = Node()
    node1.set_parent(None)
    node1.set_state(start)
    node1.set_action(get_neighbors(matriz, start, goal_number))
    node1.set_cost(0)
    node1.set_depth(0)
    frontier.append(node1)
    explored.add(start)
    
    print("\n--- INICIO DFS GRAPH SEARCH ---")
    
    while not found and frontier and iterations < iteration_limit:
        iterations += 1
        node_parent = frontier.pop()  # LIFO
        current_state = node_parent.get_state()
        
        print(f"\nIteración {iterations} - Expandiendo: {current_state}")
        print(f"  Coste: {node_parent.get_cost()}, Profundidad: {node_parent.get_depth()}")
        
        valor_actual = matriz[current_state[0]][current_state[1]][current_state[2]]
        if str(valor_actual) == goal_number:
            found = True
            node1 = node_parent
            break
        
        next_nodes = expand(node_parent, matriz, goal_number)
        
        newly_added = []
        for child in next_nodes:
            child_state = child.get_state()
            if child_state not in explored:
                explored.add(child_state)
                newly_added.append(child)
        
        frontier.extend(newly_added)
        
        print(f"  Frontera: {len(frontier)} nodos, Explorados: {len(explored)}")
    
    return print_solution(found, node1, "DFS GRAPH")


# ==============================================================================
# SECCIÓN 5: GREEDY (BEST-FIRST SEARCH)
# ==============================================================================

def expand_greedy(node_parent, matriz, goal_number, goal_position):
    """
    Expande un nodo y calcula la heurística para cada hijo.
    La heurística es la distancia Manhattan al objetivo.
    """
    children = []
    for neighbor in get_neighbors(matriz, node_parent.get_state(), goal_number):
        child = Node()
        child.set_state(neighbor)
        child.set_parent(node_parent)
        child.set_action(get_neighbors(matriz, neighbor, goal_number))
        child.set_cost(node_parent.get_cost() + 1)
        child.set_depth(node_parent.get_depth() + 1)
        
        heuristic = calculate_manhattan_distance(neighbor, goal_position)
        child.set_heuristic(heuristic)
        
        children.append(child)
    return children


def greedy_tree(matriz, start, goal_number, goal_position):
    """
    Greedy Tree Search: Búsqueda voraz que elige siempre el nodo con menor heurística.
    Usa solo h(n), no considera el coste acumulado.
    """
    frontier = []
    found = False
    iteration_limit = 1000
    iterations = 0
    
    node1 = Node()
    node1.set_parent(None)
    node1.set_state(start)
    node1.set_action(get_neighbors(matriz, start, goal_number))
    node1.set_cost(0)
    node1.set_depth(0)
    heuristic = calculate_manhattan_distance(start, goal_position)
    node1.set_heuristic(heuristic)
    frontier.append(node1)
    
    print("\n--- INICIO GREEDY TREE SEARCH ---")
    
    while not found and frontier and iterations < iteration_limit:
        iterations += 1
        
        # Ordenar por heurística (menor primero)
        frontier.sort(key=lambda n: n.get_heuristic())
        
        node_parent = frontier.pop(0)
        current_state = node_parent.get_state()
        
        print(f"\nIteración {iterations} - Expandiendo: {current_state}")
        print(f"  h(n): {node_parent.get_heuristic()}, Coste: {node_parent.get_cost()}")
        
        valor_actual = matriz[current_state[0]][current_state[1]][current_state[2]]
        if str(valor_actual) == str(goal_number):
            found = True
            node1 = node_parent
            break
        
        next_nodes = expand_greedy(node_parent, matriz, goal_number, goal_position)
        frontier.extend(next_nodes)
        
        print(f"  Frontera: {len(frontier)} nodos")
    
    return print_solution(found, node1, "GREEDY TREE")


def greedy_graph(matriz, start, goal_number, goal_position):
    """
    Greedy Graph Search: Búsqueda voraz con control de estados visitados.
    Evita ciclos y estados repetidos.
    """
    goal_number = str(goal_number)
    frontier = []
    found = False
    iteration_limit = 1000
    iterations = 0
    explored = set()
    
    node1 = Node()
    node1.set_parent(None)
    node1.set_state(start)
    node1.set_action(get_neighbors(matriz, start, goal_number))
    node1.set_cost(0)
    node1.set_depth(0)
    heuristic = calculate_manhattan_distance(start, goal_position)
    node1.set_heuristic(heuristic)
    frontier.append(node1)
    explored.add(start)
    
    print("\n--- INICIO GREEDY GRAPH SEARCH ---")
    
    while not found and frontier and iterations < iteration_limit:
        iterations += 1
        
        frontier.sort(key=lambda n: n.get_heuristic())
        
        node_parent = frontier.pop(0)
        current_state = node_parent.get_state()
        
        print(f"\nIteración {iterations} - Expandiendo: {current_state}")
        print(f"  h(n): {node_parent.get_heuristic()}, Coste: {node_parent.get_cost()}")
        
        valor_actual = matriz[current_state[0]][current_state[1]][current_state[2]]
        if str(valor_actual) == goal_number:
            found = True
            node1 = node_parent
            break
        
        next_nodes = expand_greedy(node_parent, matriz, goal_number, goal_position)
        
        for child in next_nodes:
            child_state = child.get_state()
            if child_state not in explored:
                explored.add(child_state)
                frontier.append(child)
        
        print(f"  Frontera: {len(frontier)} nodos, Explorados: {len(explored)}")
    
    return print_solution(found, node1, "GREEDY GRAPH")


# ==============================================================================
# SECCIÓN 6: A* (A-STAR)
# ==============================================================================

def expand_astar(node_parent, matriz, goal_number, goal_position):
    """
    Expande un nodo y calcula tanto el coste como la heurística para cada hijo.
    Para A*, se usa f(n) = g(n) + h(n)
    """
    children = []
    for neighbor in get_neighbors(matriz, node_parent.get_state(), goal_number):
        child = Node()
        child.set_state(neighbor)
        child.set_parent(node_parent)
        child.set_action(get_neighbors(matriz, neighbor, goal_number))
        child.set_cost(node_parent.get_cost() + 1)
        child.set_depth(node_parent.get_depth() + 1)
        
        heuristic = calculate_manhattan_distance(neighbor, goal_position)
        child.set_heuristic(heuristic)
        
        children.append(child)
    return children


def astar_tree(matriz, start, goal_number, goal_position):
    """
    A* Tree Search: Búsqueda A* que usa f(n) = g(n) + h(n).
    Combina coste real y estimado para encontrar caminos óptimos.
    """
    frontier = []
    found = False
    iteration_limit = 1000
    iterations = 0
    
    node1 = Node()
    node1.set_parent(None)
    node1.set_state(start)
    node1.set_action(get_neighbors(matriz, start, goal_number))
    node1.set_cost(0)
    node1.set_depth(0)
    heuristic = calculate_manhattan_distance(start, goal_position)
    node1.set_heuristic(heuristic)
    frontier.append(node1)
    
    print("\n--- INICIO A* TREE SEARCH ---")
    
    while not found and frontier and iterations < iteration_limit:
        iterations += 1
        
        # Ordenar por f(n) = g(n) + h(n)
        frontier.sort(key=lambda n: n.get_f_score())
        
        node_parent = frontier.pop(0)
        current_state = node_parent.get_state()
        
        print(f"\nIteración {iterations} - Expandiendo: {current_state}")
        print(f"  g(n): {node_parent.get_cost()}, h(n): {node_parent.get_heuristic()}, f(n): {node_parent.get_f_score()}")
        
        valor_actual = matriz[current_state[0]][current_state[1]][current_state[2]]
        if str(valor_actual) == str(goal_number):
            found = True
            node1 = node_parent
            break
        
        next_nodes = expand_astar(node_parent, matriz, goal_number, goal_position)
        frontier.extend(next_nodes)
        
        print(f"  Frontera: {len(frontier)} nodos")
    
    return print_solution(found, node1, "A* TREE")


def astar_graph(matriz, start, goal_number, goal_position):
    """
    A* Graph Search: Búsqueda A* con control de estados visitados.
    Garantiza encontrar el camino óptimo si la heurística es admisible.
    """
    goal_number = str(goal_number)
    frontier = []
    found = False
    iteration_limit = 1000
    iterations = 0
    explored = set()
    
    node1 = Node()
    node1.set_parent(None)
    node1.set_state(start)
    node1.set_action(get_neighbors(matriz, start, goal_number))
    node1.set_cost(0)
    node1.set_depth(0)
    heuristic = calculate_manhattan_distance(start, goal_position)
    node1.set_heuristic(heuristic)
    frontier.append(node1)
    explored.add(start)
    
    print("\n--- INICIO A* GRAPH SEARCH ---")
    
    while not found and frontier and iterations < iteration_limit:
        iterations += 1
        
        frontier.sort(key=lambda n: n.get_f_score())
        
        node_parent = frontier.pop(0)
        current_state = node_parent.get_state()
        
        print(f"\nIteración {iterations} - Expandiendo: {current_state}")
        print(f"  g(n): {node_parent.get_cost()}, h(n): {node_parent.get_heuristic()}, f(n): {node_parent.get_f_score()}")
        
        valor_actual = matriz[current_state[0]][current_state[1]][current_state[2]]
        if str(valor_actual) == goal_number:
            found = True
            node1 = node_parent
            break
        
        next_nodes = expand_astar(node_parent, matriz, goal_number, goal_position)
        
        for child in next_nodes:
            child_state = child.get_state()
            if child_state not in explored:
                explored.add(child_state)
                frontier.append(child)
        
        print(f"  Frontera: {len(frontier)} nodos, Explorados: {len(explored)}")
    
    return print_solution(found, node1, "A* GRAPH")


# ==============================================================================
# SECCIÓN 7: CASOS DE USO Y EJEMPLOS
# ==============================================================================

def ejecutar_ejemplos():
    """Ejecuta varios ejemplos de búsqueda con diferentes algoritmos"""
    
    print("\n" + "="*80)
    print("EJEMPLOS DE BÚSQUEDA EN CENTRO DE DATOS")
    print("="*80)
    
    # Ejemplo 1: BFS - Del servidor 55 al 43
    print("\n" + "-"*80)
    print("EJEMPLO 1: BFS TREE - Servidor 55 (nivel 0) → Servidor 43 (nivel 0)")
    print("-"*80)
    origen1 = Lista_Origenes['55']
    destino1 = Lista_Destinos['43']
    bfs_tree(matriz, origen1, '43')
    
    # Ejemplo 2: BFS GRAPH - Del servidor 30 al 58
    print("\n" + "-"*80)
    print("EJEMPLO 2: BFS GRAPH - Servidor 30 (nivel 0) → Servidor 58 (nivel 0)")
    print("-"*80)
    origen2 = Lista_Origenes['30']
    destino2 = Lista_Destinos['58']
    bfs_graph(matriz, origen2, '58')
    
    # Ejemplo 3: DFS TREE - Del servidor 07 al 60
    print("\n" + "-"*80)
    print("EJEMPLO 3: DFS TREE - Servidor 07 (nivel 0) → Servidor 60 (nivel 0)")
    print("-"*80)
    origen3 = Lista_Origenes['07']
    destino3 = Lista_Destinos['60']
    dfs_tree(matriz, origen3, '60')
    
    # Ejemplo 4: DFS GRAPH - Del servidor 145 al 150
    print("\n" + "-"*80)
    print("EJEMPLO 4: DFS GRAPH - Servidor 145 (nivel 1) → Servidor 150 (nivel 1)")
    print("-"*80)
    origen4 = Lista_Origenes['145']
    destino4 = Lista_Destinos['150']
    dfs_graph(matriz, origen4, '150')
    
    # Ejemplo 5: GREEDY TREE - Del servidor 26 al 116
    print("\n" + "-"*80)
    print("EJEMPLO 5: GREEDY TREE - Servidor 26 (nivel 0) → Servidor 116 (nivel 1)")
    print("-"*80)
    origen5 = Lista_Origenes['26']
    destino5 = Lista_Destinos['116']
    greedy_tree(matriz, origen5, '116', destino5)
    
    # Ejemplo 6: GREEDY GRAPH - Del servidor 78 al 148
    print("\n" + "-"*80)
    print("EJEMPLO 6: GREEDY GRAPH - Servidor 78 (nivel 0) → Servidor 148 (nivel 1)")
    print("-"*80)
    origen6 = Lista_Origenes['78']
    destino6 = Lista_Destinos['148']
    greedy_graph(matriz, origen6, '148', destino6)
    
    # Ejemplo 7: A* TREE - Del servidor 194 al 159
    print("\n" + "-"*80)
    print("EJEMPLO 7: A* TREE - Servidor 194 (nivel 1) → Servidor 159 (nivel 1)")
    print("-"*80)
    origen7 = Lista_Origenes['194']
    destino7 = Lista_Destinos['159']
    astar_tree(matriz, origen7, '159', destino7)
    
    # Ejemplo 8: A* GRAPH - Del servidor 55 al 150
    print("\n" + "-"*80)
    print("EJEMPLO 8: A* GRAPH - Servidor 55 (nivel 0) → Servidor 150 (nivel 1)")
    print("-"*80)
    origen8 = Lista_Origenes['55']
    destino8 = Lista_Destinos['150']
    astar_graph(matriz, origen8, '150', destino8)


# ==============================================================================
# EJECUCIÓN PRINCIPAL
# ==============================================================================

if __name__ == "__main__":
    ejecutar_ejemplos()
