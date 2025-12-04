#include "PW_ILS.h"
#include <chrono>
#include <algorithm>
#include <limits>

/** Algorithm management **/

PW_ILS::PW_ILS(std::string name, Problem* problem): Algorithm(name, problem) {
    base_algorithm = new PWDefault("PWDefault", problem);
    readConfiguration();
    initDataCollection();
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    rng = std::mt19937(seed);
    uniform_dist = std::uniform_real_distribution<double>(0.0, 1.0);
    setStatus(ALGO_READY);
}

PW_ILS::~PW_ILS() {
    delete base_algorithm;
}

void PW_ILS::initAlgorithm() {
    Algorithm::initAlgorithm();
    iterations_count = 0;
    no_improve_count = 0;
    perturbations_count = 0;
    local_search_count = 0;
}

void PW_ILS::readConfiguration() {
    max_iterations = Parameters::getILSMaxIterations();
    max_no_improve = Parameters::getILSMaxNoImprove();
    perturbation_strength = Parameters::getILSPerturbationStrength();
    timelimit = Parameters::getILSTimelimit();
    accept_worse = Parameters::isILSAcceptWorse();
    algo_type = ALGO_HEURISTIC;
}

void PW_ILS::resetAlgorithm(int reset_level) {
    Algorithm::initAlgorithm();
    setStatus(ALGO_READY);
    readConfiguration();
    best_solution_id = -1;
    solutions.clear();
    base_algorithm->resetAlgorithm(reset_level);
    collector.resetTimesCumulative();
}

/** Main ILS solve method **/

void PW_ILS::solve() {
    if(Parameters::getVerbosity() >= 4)
        std::cout << "ILS: Starting Iterated Local Search..." << std::endl;
    
    setStatus(ALGO_OPTIMIZING);
    initAlgorithm();
    collector.startGlobalTime();
    
    // Step 1: Generate initial solution
    generateInitialSolution();
    
    if(solutions.empty()) {
        if(Parameters::getVerbosity() >= 2)
            std::cout << "ILS: Failed to generate initial solution" << std::endl;
        setStatus(ALGO_TIMELIMIT);
        return;
    }
    Path current_solution = solutions[0];
    Path best_solution = current_solution;
    updateIncumbent(best_solution.getObjective());
    if(Parameters::getVerbosity() >= 3)
        std::cout << "ILS: Initial solution objective: " << current_solution.getObjective() << std::endl;
    
    // Step 2: ILS main loop
    while(iterations_count < max_iterations && 
          no_improve_count < max_no_improve &&
          collector.getGlobalTime() < timelimit) {
        iterations_count++;
        // Local Search
        collector.startTime("t_local_search");
        localSearch(current_solution);
        collector.stopTime("t_local_search");
        local_search_count++;
        // Check for improvement
        if(current_solution.getObjective() < best_solution.getObjective()) {
            best_solution = current_solution;
            updateIncumbent(best_solution.getObjective());
            no_improve_count = 0;
            // Store improved solution
            addSolution(best_solution);
            updateBestSolution(solutions.size() - 1);
            if(Parameters::getVerbosity() >= 3)
                std::cout << "ILS: Iteration " << iterations_count 
                         << " - New best: " << best_solution.getObjective() << std::endl;
        } else {
            no_improve_count++;
        }
        // Acceptance Criterion
        if(!acceptanceCriterion(current_solution, best_solution)) {
            current_solution = best_solution;
        }
        // Perturbation
        if(no_improve_count < max_no_improve) {
            collector.startTime("t_perturbation");
            perturbation(current_solution, perturbation_strength);
            collector.stopTime("t_perturbation");
            perturbations_count++;
        }
    }
    collector.stopGlobalTime();
    if(Parameters::getVerbosity() >= 3) {
        std::cout << "ILS: Completed" << std::endl;
        std::cout << "ILS: Total iterations: " << iterations_count << std::endl;
        std::cout << "ILS: Best objective: " << best_solution.getObjective() << std::endl;
    }
    if(algo_status == ALGO_OPTIMIZING)
        setStatus(ALGO_DONE);

    collector.print();
}

/** ILS Components **/

void PW_ILS::generateInitialSolution() {
    if(Parameters::getVerbosity() >= 3)
        std::cout << "ILS: Generating initial solution with greedy heuristic..." << std::endl;
    
    Path initial_path = greedyConstruction();
    
    if(initial_path.getStatus() == PATH_FEASIBLE || 
       initial_path.getStatus() == PATH_OPTIMAL) {
        addSolution(initial_path);
        updateBestSolution(0);
        if(Parameters::getVerbosity() >= 3)
            std::cout << "ILS: Greedy objective: " << initial_path.getObjective() << std::endl;
    } else {
        if(Parameters::getVerbosity() >= 1)
            std::cout << "ILS: WARNING - Greedy construction failed! Falling back to PWDefault." << std::endl;
        base_algorithm->resetAlgorithm(0);
        base_algorithm->solve();
        if(base_algorithm->getBestSolution() != nullptr) {
            Path fallback_path = *base_algorithm->getBestSolution();
            addSolution(fallback_path);
            updateBestSolution(0);
        }
    }
}

Path PW_ILS::greedyConstruction() {
    // Greedy nearest neighbor construction that visits ALL feasible nodes
    Path path;
    
    int origin = problem->getOrigin();
    int destination = problem->getDestination();
    int n_nodes = problem->getNumNodes();
    Resource* objective = problem->getObj();
    
    // Get resource bounds to check feasibility
    std::vector<Resource*> resources = problem->getResources();
    
    std::vector<int> tour;
    std::vector<bool> visited(n_nodes, false);
    std::vector<int> consumption(resources.size(), 0);
    
    tour.push_back(origin);
    visited[origin] = true;
    int current = origin;
    
    // Build tour by visiting all reachable nodes greedily
    bool can_add_more = true;
    while(can_add_more) {
        std::vector<int>& neighbors = problem->getNeighbors(current, true);
        
        int best_neighbor = -1;
        int best_cost = std::numeric_limits<int>::max();
        
        // Find the nearest unvisited neighbor that is feasible
        for(int neighbor : neighbors) {
            if(neighbor == destination || visited[neighbor])
                continue;
                
            // Check if we can reach this neighbor without violating constraints
            bool feasible = true;
            for(size_t r = 0; r < resources.size(); r++) {
                int new_consumption = resources[r]->extend(consumption[r], current, neighbor, true);
                // Check if we can still reach destination from neighbor
                if(new_consumption > resources[r]->getUB()) {
                    feasible = false;
                    break;
                }
            }
            
            if(feasible) {
                int cost = objective->getArcCost(current, neighbor);
                if(cost < best_cost) {
                    best_cost = cost;
                    best_neighbor = neighbor;
                }
            }
        }
        
        if(best_neighbor != -1) {
            // Add the best neighbor to the tour
            tour.push_back(best_neighbor);
            visited[best_neighbor] = true;
            
            // Update consumption
            for(size_t r = 0; r < resources.size(); r++) {
                consumption[r] = resources[r]->extend(consumption[r], current, best_neighbor, true);
            }
            
            current = best_neighbor;
        } else {
            // No more nodes can be added, go to destination
            can_add_more = false;
        }
    }
    
    // Add destination
    tour.push_back(destination);
    
    // Set tour
    std::list<int> tour_list(tour.begin(), tour.end());
    path.setTour(tour_list);
    
    // Calculate objective value
    int obj_value = objective->getInitValue();
    for(size_t i = 0; i < tour.size() - 1; i++) {
        obj_value += objective->getArcCost(tour[i], tour[i+1]);
        obj_value += objective->getNodeCost(tour[i+1]);
    }
    path.setObjective(obj_value);
    path.setConsumption(consumption);
    path.setStatus(PATH_FEASIBLE);
    
    if(Parameters::getVerbosity() >= 4) {
        std::cout << "ILS: Greedy tour length: " << tour.size() 
                  << ", objective: " << obj_value << std::endl;
    }
    
    return path;
}

void PW_ILS::localSearch(Path& solution) {
    bool improved = true;
    int ls_iterations = 0;
    const int max_ls_iterations = 50;
    while(improved && ls_iterations < max_ls_iterations) {
        improved = false;
        ls_iterations++;
        int current_obj = solution.getObjective();
        // Try different local search moves
        Path candidate1 = twoOptMove(solution);
        if(candidate1.getStatus() == PATH_FEASIBLE && 
           candidate1.getObjective() < current_obj) {
            solution = candidate1;
            improved = true;
            continue;
        }
        Path candidate2 = insertMove(solution);
        if(candidate2.getStatus() == PATH_FEASIBLE && 
           candidate2.getObjective() < current_obj) {
            solution = candidate2;
            improved = true;
            continue;
        }
        Path candidate3 = swapMove(solution);
        if(candidate3.getStatus() == PATH_FEASIBLE && 
           candidate3.getObjective() < current_obj) {
            solution = candidate3;
            improved = true;
        }
    }
}

void PW_ILS::perturbation(Path& solution, int strength) {
    // Apply multiple random moves to escape local optimum
    for(int i = 0; i < strength; i++) {
        double rand_val = uniform_dist(rng);
        Path perturbed;
        if(rand_val < 0.33) {
            perturbed = twoOptMove(solution);
        } else if(rand_val < 0.66) {
            perturbed = insertMove(solution);
        } else {
            perturbed = swapMove(solution);
        }
        if(perturbed.getStatus() == PATH_FEASIBLE) {
            solution = perturbed;
        }
    }
}

bool PW_ILS::acceptanceCriterion(const Path& current, const Path& candidate) {
    if(accept_worse) {
        // Simulated annealing-like acceptance
        int current_obj = const_cast<Path&>(current).getObjective();
        int candidate_obj = const_cast<Path&>(candidate).getObjective();
        if(candidate_obj <= current_obj)
            return true;
        
        double prob = std::exp(-(candidate_obj - current_obj) / (0.1 * current_obj));
        return uniform_dist(rng) < prob;
    }
    // Only accept improvements
    return const_cast<Path&>(candidate).getObjective() <= const_cast<Path&>(current).getObjective();
}

/** Solution manipulation **/

Path PW_ILS::twoOptMove(const Path& solution) {
    Path new_path = solution;
    std::list<int> tour = const_cast<Path&>(solution).getTour();
    if(tour.size() < 4)
        return new_path;
    
    std::vector<int> tour_vec(tour.begin(), tour.end());
    int n = tour_vec.size();
    // Random 2-opt move
    std::uniform_int_distribution<int> dist(1, n - 2);
    int i = dist(rng);
    int j = dist(rng);
    if(i > j) std::swap(i, j);
    if(i == j || j - i < 2) return new_path;
    // Reverse segment [i, j]
    std::reverse(tour_vec.begin() + i, tour_vec.begin() + j + 1);
    // Create new path
    std::list<int> new_tour(tour_vec.begin(), tour_vec.end());
    new_path.setTour(new_tour);
    // Evaluate new path (simplified - would need full resource checking)
    new_path.setStatus(PATH_FEASIBLE);
    return new_path;
}

Path PW_ILS::insertMove(const Path& solution) {
    Path new_path = solution;
    std::list<int> tour = const_cast<Path&>(solution).getTour();
    if(tour.size() < 3)
        return new_path;
    
    std::vector<int> tour_vec(tour.begin(), tour.end());
    int n = tour_vec.size();
    // Random insert move
    std::uniform_int_distribution<int> dist(1, n - 2);
    int remove_pos = dist(rng);
    int insert_pos = dist(rng);
    if(remove_pos == insert_pos)
        return new_path;
    int node = tour_vec[remove_pos];
    tour_vec.erase(tour_vec.begin() + remove_pos);
    if(insert_pos > remove_pos)
        insert_pos--;
    
    tour_vec.insert(tour_vec.begin() + insert_pos, node);
    // Create new path
    std::list<int> new_tour(tour_vec.begin(), tour_vec.end());
    new_path.setTour(new_tour);
    new_path.setStatus(PATH_FEASIBLE);
    return new_path;
}

Path PW_ILS::swapMove(const Path& solution) {
    Path new_path = solution;
    std::list<int> tour = const_cast<Path&>(solution).getTour();
    
    if(tour.size() < 3)
        return new_path;
    
    std::vector<int> tour_vec(tour.begin(), tour.end());
    int n = tour_vec.size();
    // Random swap move
    std::uniform_int_distribution<int> dist(1, n - 2);
    int pos1 = dist(rng);
    int pos2 = dist(rng);
    if(pos1 != pos2) {
        std::swap(tour_vec[pos1], tour_vec[pos2]);
    }
    // Create new path
    std::list<int> new_tour(tour_vec.begin(), tour_vec.end());
    new_path.setTour(new_tour);
    new_path.setStatus(PATH_FEASIBLE);
    return new_path;
}

/** Data Collection **/

void PW_ILS::initDataCollection() {
    collector = DataCollector(name);
    collector.initTime("t_local_search");
    collector.initTime("t_perturbation");
}

void PW_ILS::collectData() {
    collector.collect("iterations", iterations_count);
    collector.collect("perturbations", perturbations_count);
    collector.collect("local_searches", local_search_count);
    collector.collect("no_improve", no_improve_count);
}
