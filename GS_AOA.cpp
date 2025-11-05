// #include <bits/stdc++.h>
// using namespace std;

// // Parameters
// const int HOURS = 24;
// const int POP_SIZE = 10;
// const int GENERATIONS = 30;
// const double CROSSOVER_RATE = 0.8;
// const double MUTATION_RATE = 0.2;

// // Tariff (0-5=1, 6-17=2, 18-21=5, 22-23=1)
// vector<double> tariff = {1,1,1,1,1,1,
//                          2,2,2,2,2,2,2,2,2,2,2,2,
//                          5,5,5,5,
//                          1,1};

// // Appliance structure
// struct Appliance {
//     string name;
//     int duration;
//     double power;
// };

// vector<Appliance> appliances = {
//     {"Washing Machine", 2, 1.0},
//     {"Dishwasher", 1, 0.8},
//     {"Water Heater", 1, 1.5}
// };

// // Chromosome = vector of schedules for each appliance
// using Chromosome = vector<vector<int>>;

// // Generate random chromosome
// Chromosome createChromosome() {
//     Chromosome chromosome;
//     for (auto &app : appliances) {
//         int start = rand() % (HOURS - app.duration + 1); 
//         vector<int> schedule(HOURS, 0);
//         for (int i = start; i < start + app.duration; i++) 
//             schedule[i] = 1;
//         chromosome.push_back(schedule);
//     }
//     return chromosome;
// }

// // Fitness function (lower = better)
// double fitness(const Chromosome &chromosome) {
//     double total_cost = 0;
//     for (int a = 0; a < appliances.size(); a++) {
//         for (int h = 0; h < HOURS; h++) {
//             total_cost += chromosome[a][h] * appliances[a].power * tariff[h];
//         }
//     }
//     return total_cost;
// }

// // Roulette wheel selection
// Chromosome selection(const vector<Chromosome> &population) {
//     vector<double> weights;
//     for (auto &ch : population)
//         weights.push_back(1.0 / (fitness(ch) + 1e-6));

//     double sum = accumulate(weights.begin(), weights.end(), 0.0);
//     double pick = ((double) rand() / RAND_MAX) * sum;
//     double current = 0;

//     for (int i = 0; i < population.size(); i++) {
//         current += weights[i];
//         if (current >= pick)
//             return population[i];
//     }
//     return population.back();
// }

// // Crossover
// pair<Chromosome, Chromosome> crossover(const Chromosome &p1, const Chromosome &p2) {
//     if (((double) rand() / RAND_MAX) < CROSSOVER_RATE) {
//         int point = rand() % appliances.size();
//         Chromosome c1, c2;
//         for (int i = 0; i < appliances.size(); i++) {
//             if (i < point) {
//                 c1.push_back(p1[i]);
//                 c2.push_back(p2[i]);
//             } else {
//                 c1.push_back(p2[i]);
//                 c2.push_back(p1[i]);
//             }
//         }
//         return {c1, c2};
//     }
//     return {p1, p2};
// }

// // Mutation (shift appliance randomly)
// Chromosome mutate(Chromosome ch) {
//     if (((double) rand() / RAND_MAX) < MUTATION_RATE) {
//         int idx = rand() % appliances.size();  
//         int duration = appliances[idx].duration;
//         vector<int> schedule(HOURS, 0);
//         int start = rand() % (HOURS - duration + 1);
//         for (int i = start; i < start + duration; i++)
//             schedule[i] = 1;
//         ch[idx] = schedule;
//     }
//     return ch;
// }

// // Genetic Algorithm
// Chromosome geneticAlgorithm() {
//     vector<Chromosome> population;
//     for (int i = 0; i < POP_SIZE; i++)
//         population.push_back(createChromosome());

//     Chromosome best = population[0];
//     for (int gen = 0; gen < GENERATIONS; gen++) {
//         vector<Chromosome> new_population;
//         for (int i = 0; i < POP_SIZE / 2; i++) {
//             Chromosome parent1 = selection(population);
//             Chromosome parent2 = selection(population);
//             pair<Chromosome, Chromosome> children = crossover(parent1, parent2);
//                 Chromosome child1 = mutate(children.first);
//                 Chromosome child2 = mutate(children.second);
//                 new_population.push_back(child1);
//                 new_population.push_back(child2);

//         }
//         population = new_population;

//         for (auto &ch : population)
//             if (fitness(ch) < fitness(best))
//                 best = ch;

//         cout << "Gen " << gen+1 << ": Best Cost = " << fitness(best) << endl;
//     }
//     return best;
// }

// int main() {
//     srand(time(0));
//     Chromosome best = geneticAlgorithm();

//     cout << "\nBest Schedule:\n";
//     for (int a = 0; a < appliances.size(); a++) {
//         cout << appliances[a].name << ": ";
//         for (int h = 0; h < HOURS; h++) {
//             if (best[a][h] == 1) cout << h << " ";
//         }
//         cout << endl;
//     }
//     return 0;
// }


#include <bits/stdc++.h>
using namespace std;

// ---------- Problem setup (same as your GA) ----------
const int HOURS = 24;

// Tariff (0-5=1, 6-17=2, 18-21=5, 22-23=1)
vector<double> tariff = {
    1,1,1,1,1,1,
    2,2,2,2,2,2,2,2,2,2,2,2,
    5,5,5,5,
    1,1
};

struct Appliance {
    string name;
    int duration;   // contiguous hours needed
    double power;   // kW
};

vector<Appliance> appliances = {
    {"Washing Machine", 2, 1.0},
    {"Dishwasher",      1, 0.8},
    {"Water Heater",    1, 1.5}
};

// ---------- AOA parameters (tune like GA params) ----------
const int POP_SIZE   = 20;    // number of objects (solutions)
const int ITERATIONS = 60;    // iterations (generations)
const double C1 = 2.0;
const double C2 = 6.0;
const double C3 = 2.0;
const double C4 = 1.0;
const double U  = 0.9;        // acceleration normalization upper
const double L  = 0.1;        // acceleration normalization lower

// ---------- Helper: bounds for each decision (start hour per appliance) ----------
struct Bounds { int lo, hi; }; // inclusive bounds on integer start
vector<Bounds> varBounds() {
    vector<Bounds> b;
    for (auto &ap : appliances) {
        b.push_back({0, HOURS - ap.duration});
    }
    return b;
}

// ---------- Evaluate cost from integer starts ----------
double fitness_from_starts(const vector<int> &start) {
    double cost = 0.0;
    for (int a = 0; a < (int)appliances.size(); a++) {
        int s = start[a];
        int d = appliances[a].duration;
        double p = appliances[a].power;
        for (int h = s; h < s + d; ++h) cost += p * tariff[h];
    }
    return cost; // lower is better
}

// ---------- Clamp/round continuous -> valid integer starts ----------
void discretize_and_clamp(const vector<double> &x, vector<int> &start,
                          const vector<Bounds> &B) {
    start.resize(x.size());
    for (int i = 0; i < (int)x.size(); ++i) {
        int v = (int)round(x[i]);
        if (v < B[i].lo) v = B[i].lo;
        if (v > B[i].hi) v = B[i].hi;
        start[i] = v;
    }
}

// ---------- Random in [0,1] ----------
inline double urand() { return (double)rand() / (double)RAND_MAX; }

// ---------- Build schedule print like GA (from integer starts) ----------
void print_schedule(const vector<int> &start) {
    for (int a = 0; a < (int)appliances.size(); a++) {
        cout << appliances[a].name << ": ";
        for (int h = start[a]; h < start[a] + appliances[a].duration; ++h) {
            cout << h << " ";
        }
        cout << "\n";
    }
}

// =====================================================
//               Archimedes Optimization
// =====================================================
// We keep continuous state for AOA (x, den, vol, acc) per decision variable.
// Decision variables = #appliances (each = start hour).
// For fitness we discretize x -> int start and compute cost.

int main() {
    srand((unsigned)time(0));

    const int D = (int)appliances.size();     // #decision variables
    const vector<Bounds> B = varBounds();     // per-var bounds

    // --- Population state ---
    vector<vector<double>> X(POP_SIZE, vector<double>(D));     // positions
    vector<vector<double>> DEN(POP_SIZE, vector<double>(D));   // densities
    vector<vector<double>> VOL(POP_SIZE, vector<double>(D));   // volumes
    vector<vector<double>> ACC(POP_SIZE, vector<double>(D));   // accelerations

    // --- Init population uniformly inside bounds ---
    auto rand_in_bounds = [&](int i) {
        return B[i].lo + urand() * (B[i].hi - B[i].lo);
    };

    for (int i = 0; i < POP_SIZE; ++i) {
        for (int d = 0; d < D; ++d) {
            X[i][d]   = rand_in_bounds(d);
            DEN[i][d] = urand();
            VOL[i][d] = urand();
            ACC[i][d] = urand();
        }
    }

    // --- Evaluate initial best ---
    double bestCost = 1e100;
    vector<double> Xbest(D);
    vector<double> DENbest(D), VOLbest(D), ACCbest(D);

    for (int i = 0; i < POP_SIZE; ++i) {
        vector<int> s;
        discretize_and_clamp(X[i], s, B);
        double f = fitness_from_starts(s);
        if (f < bestCost) {
            bestCost = f;
            Xbest = X[i];
            DENbest = DEN[i];
            VOLbest = VOL[i];
            ACCbest = ACC[i];
        }
    }

    // --- Work buffers ---
    vector<vector<double>> DENn(POP_SIZE, vector<double>(D));
    vector<vector<double>> VOLn(POP_SIZE, vector<double>(D));
    vector<vector<double>> ACCn(POP_SIZE, vector<double>(D));
    vector<vector<double>> ACCnorm(POP_SIZE, vector<double>(D));

    // --- Main loop ---
    for (int t = 1; t <= ITERATIONS; ++t) {

        // 1) Update density/volume towards best
        for (int i = 0; i < POP_SIZE; ++i) {
            for (int d = 0; d < D; ++d) {
                DENn[i][d] = DEN[i][d] + urand() * (DENbest[d] - DEN[i][d]);
                VOLn[i][d] = VOL[i][d] + urand() * (VOLbest[d] - VOL[i][d]);
            }
        }

        // 2) Transfer operator TF and density decreasing factor "d"
        double TF = exp((double(t) - ITERATIONS) / (double)ITERATIONS);
        double dens_decay = exp((double(ITERATIONS) - t) / (double)ITERATIONS) - (double)t/(double)ITERATIONS;

        // 3) Update acceleration (explore vs exploit)
        for (int i = 0; i < POP_SIZE; ++i) {
            if (TF <= 0.5) {
                // Exploration: collide with random material
                int mr = rand() % POP_SIZE;
                for (int d = 0; d < D; ++d) {
                    double num  = DEN[mr][d] + VOL[mr][d] * ACC[mr][d];
                    double deno = DENn[i][d] + VOLn[i][d];
                    ACCn[i][d] = num / (deno + 1e-12);
                }
            } else {
                // Exploitation: use the best
                for (int d = 0; d < D; ++d) {
                    double num  = DENbest[d] + VOLbest[d] * ACCbest[d];
                    double deno = DENn[i][d] + VOLn[i][d];
                    ACCn[i][d] = num / (deno + 1e-12);
                }
            }
        }

        // 4) Normalize acceleration per dimension across pop
        for (int d = 0; d < D; ++d) {
            double mn = 1e100, mx = -1e100;
            for (int i = 0; i < POP_SIZE; ++i) {
                if (ACCn[i][d] < mn) mn = ACCn[i][d];
                if (ACCn[i][d] > mx) mx = ACCn[i][d];
            }
            double denom = (mx - mn) + 1e-12;
            for (int i = 0; i < POP_SIZE; ++i) {
                double z = (ACCn[i][d] - mn) / denom;
                ACCnorm[i][d] = U * z + L; // in [L,U]
            }
        }

        // 5) Update positions (explore/exploit)
        for (int i = 0; i < POP_SIZE; ++i) {
            if (TF <= 0.5) {
                // exploration
                int rr = rand() % POP_SIZE;
                for (int d = 0; d < D; ++d) {
                    double step = C1 * urand() * ACCnorm[i][d] * dens_decay * (X[rr][d] - X[i][d]);
                    X[i][d] += step;

                    // clamp to per-var bounds
                    if (X[i][d] < B[d].lo) X[i][d] = B[d].lo;
                    if (X[i][d] > B[d].hi) X[i][d] = B[d].hi;
                }
            } else {
                // exploitation
                double Tpar = C3 * TF;
                double P = 2.0 * urand() - C4;
                double F = (P <= 0.5) ? +1.0 : -1.0;

                for (int d = 0; d < D; ++d) {
                    double step = F * C2 * urand() * ACCnorm[i][d] * dens_decay * (Tpar * Xbest[d] - X[i][d]);
                    X[i][d] = Xbest[d] + step;

                    if (X[i][d] < B[d].lo) X[i][d] = B[d].lo;
                    if (X[i][d] > B[d].hi) X[i][d] = B[d].hi;
                }
            }
        }

        // 6) Commit new DEN/VOL/ACC, evaluate, update best
        DEN.swap(DENn);
        VOL.swap(VOLn);
        ACC.swap(ACCn);

        for (int i = 0; i < POP_SIZE; ++i) {
            vector<int> s;
            discretize_and_clamp(X[i], s, B);
            double f = fitness_from_starts(s);
            if (f < bestCost) {
                bestCost = f;
                Xbest = X[i];
                DENbest = DEN[i];
                VOLbest = VOL[i];
                ACCbest = ACC[i];
            }
        }

        cout << "Iter " << t << ": Best Cost = " << bestCost << "\n";
    }

    // --- Final best schedule (rounded) ---
    vector<int> bestStart;
    discretize_and_clamp(Xbest, bestStart, B);

    cout << "\nBest Schedule:\n";
    print_schedule(bestStart);
    cout << "\nMinimum Electricity Cost = " << bestCost << "\n";

    return 0;
}

//Pricing scheme all
//1.)pAR(peak average ratio)
