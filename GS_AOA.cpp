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

const int HOURS = 24;
const int POP_SIZE = 10;
const int MAX_ITERS = 30;

// Tariff rates
vector<double> tariff = {1,1,1,1,1,1,
                         2,2,2,2,2,2,2,2,2,2,2,2,
                         5,5,5,5,
                         1,1};

struct Appliance {
    string name;
    int duration;
    double power;
};

vector<Appliance> appliances = {
    {"Washing Machine", 2, 1.0},
    {"Dishwasher", 1, 0.8},
    {"Water Heater", 1, 1.5}
};

// Chromosome = vector of schedules
using Chromosome = vector<vector<int>>;

struct Object {
    Chromosome schedule;
    double fitness;
    double density, volume, acc;
};

// Generate random schedule
Chromosome createChromosome() {
    Chromosome chromosome;
    for (auto &app : appliances) {
        int start = rand() % (HOURS - app.duration + 1);
        vector<int> schedule(HOURS, 0);
        for (int i = start; i < start + app.duration; i++)
            schedule[i] = 1;
        chromosome.push_back(schedule);
    }
    return chromosome;
}

// Fitness = electricity cost
double fitness(const Chromosome &chromosome) {
    double total_cost = 0;
    for (int a = 0; a < appliances.size(); a++) {
        for (int h = 0; h < HOURS; h++) {
            total_cost += chromosome[a][h] * appliances[a].power * tariff[h];
        }
    }
    return total_cost;
}

// AOA main algorithm
Chromosome AOA() {
    vector<Object> population(POP_SIZE);
    for (int i = 0; i < POP_SIZE; i++) {
        population[i].schedule = createChromosome();
        population[i].fitness = fitness(population[i].schedule);
        population[i].density = ((double) rand() / RAND_MAX);
        population[i].volume = ((double) rand() / RAND_MAX);
        population[i].acc = ((double) rand() / RAND_MAX);
    }

    Object best = population[0];
    for (auto &obj : population)
        if (obj.fitness < best.fitness) best = obj;

    for (int t = 1; t <= MAX_ITERS; t++) {
        double T = exp(-((double)t / MAX_ITERS)); // transfer factor
        for (int i = 0; i < POP_SIZE; i++) {
            // Update density, volume, acceleration
            population[i].density += ((double) rand() / RAND_MAX) * (best.density - population[i].density);
            population[i].volume  += ((double) rand() / RAND_MAX) * (best.volume - population[i].volume);
            population[i].acc     += ((double) rand() / RAND_MAX) * (best.acc - population[i].acc);

            Chromosome new_schedule = population[i].schedule;

            // Move appliances
            for (int a = 0; a < appliances.size(); a++) {
                if (((double) rand() / RAND_MAX) < T) {
                    // Exploitation: move closer to best
                    new_schedule[a] = best.schedule[a];
                } else {
                    // Exploration: random shift
                    int start = rand() % (HOURS - appliances[a].duration + 1);
                    vector<int> schedule(HOURS, 0);
                    for (int h = start; h < start + appliances[a].duration; h++)
                        schedule[h] = 1;
                    new_schedule[a] = schedule;
                }
            }

            double new_fit = fitness(new_schedule);
            if (new_fit < population[i].fitness) {
                population[i].schedule = new_schedule;
                population[i].fitness = new_fit;
            }

            if (population[i].fitness < best.fitness)
                best = population[i];
        }
        cout << "Iter " << t << ": Best Cost = " << best.fitness << endl;
    }

    return best.schedule;
}

int main() {
    srand(time(0));
    Chromosome best = AOA();

    cout << "\nBest Schedule (AOA):\n";
    for (int a = 0; a < appliances.size(); a++) {
        cout << appliances[a].name << ": ";
        for (int h = 0; h < HOURS; h++) {
            if (best[a][h] == 1) cout << h << " ";
        }
        cout << endl;
    }
    return 0;
}
//Pricing scheme all
//1.)pAR(peak average ratio)
