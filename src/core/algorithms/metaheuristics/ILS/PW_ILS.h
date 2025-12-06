#ifndef PW_ILS_H
#define PW_ILS_H

#include "algorithms/algorithm.h"
#include "algorithms/dynamic_programming/PW_default/PW_default.h"
#include <random>

class PW_ILS: public Algorithm {

public:

    /** Algorithm management **/
    PW_ILS(std::string name, Problem* problem);
    ~PW_ILS();

    void initAlgorithm();
    void readConfiguration();
    void resetAlgorithm(int reset_level);

    void solve() override;

    /** ILS Components **/
    void generateInitialSolution();
    Path greedyConstruction();
    void localSearch(Path& solution);
    void perturbation(Path& solution, int strength);
    bool acceptanceCriterion(const Path& current, const Path& candidate);

    /** Solution manipulation **/
    void evaluatePath(Path& path);
    Path addNodeMove(const Path& solution);
    Path removeNodeMove(const Path& solution);
    Path twoOptMove(const Path& solution);
    Path insertMove(const Path& solution);
    Path swapMove(const Path& solution);

    /** Data Collection **/
    void initDataCollection();
    void collectData();

private:

    //Base algorithm for constructing solutions
    PWDefault* base_algorithm;

    //ILS Parameters
    int max_iterations;
    int max_no_improve;
    int perturbation_strength;
    float timelimit;
    bool accept_worse;

    //Random number generation
    std::mt19937 rng;
    std::uniform_real_distribution<double> uniform_dist;

    //Statistics
    int iterations_count;
    int no_improve_count;
    int perturbations_count;
    int local_search_count;
};

#endif //PW_ILS_H
