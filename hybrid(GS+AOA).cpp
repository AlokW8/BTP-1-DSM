#include <bits/stdc++.h>
using namespace std;

// ========================= Common Problem Setup =========================
const int HOURS = 24;

// Default tariff (you can override from stdin at runtime)
vector<double> tariff_default = {
    1,1,1,1,1,1,          // 0-5
    2,2,2,2,2,2,2,2,2,2,2,2, // 6-17
    5,5,5,5,              // 18-21
    1,1                   // 22-23
};

struct Appliance {
    string name;
    int duration;   // contiguous hours
    double power;   // kW
};

vector<Appliance> appliances = {
    {"Washing Machine", 2, 1.0},
    {"Dishwasher",      1, 0.8},
    {"Water Heater",    1, 1.5}
};

// Cost from integer starts
double cost_from_starts(const vector<int>& start, const vector<double>& tariff) {
    double cost = 0.0;
    for (int a = 0; a < (int)appliances.size(); a++) {
        int s = start[a];
        int d = appliances[a].duration;
        double p = appliances[a].power;
        for (int h = s; h < s + d; ++h) cost += p * tariff[h];
    }
    return cost;
}

// Utility: print schedule given starts
void print_schedule(const vector<int>& start) {
    for (int a = 0; a < (int)appliances.size(); a++) {
        cout << appliances[a].name << ": ";
        for (int h = start[a]; h < start[a] + appliances[a].duration; ++h) {
            cout << h << " ";
        }
        cout << "\n";
    }
}

// ========================= GA (your style, lightly adapted) =========================
const int GA_POP_SIZE   = 10;
const int GA_GENERATIONS= 30;
const double CROSSOVER_RATE = 0.8;
const double MUTATION_RATE  = 0.2;

using Chromosome = vector<vector<int>>; // per-appliance 24-length 0/1 schedule

// Build schedule array (0/1) from a start time & duration
vector<int> schedule_from_start(int start, int dur) {
    vector<int> v(HOURS, 0);
    for (int i = start; i < start + dur; ++i) v[i] = 1;
    return v;
}

// Create random valid chromosome (per-appliance contiguous block)
Chromosome ga_createChromosome() {
    Chromosome chromosome;
    for (auto &app : appliances) {
        int start = rand() % (HOURS - app.duration + 1);
        chromosome.push_back(schedule_from_start(start, app.duration));
    }
    return chromosome;
}

// Fitness for GA (lower is better)
double ga_fitness(const Chromosome &chromosome, const vector<double>& tariff) {
    double total_cost = 0.0;
    for (int a = 0; a < (int)appliances.size(); a++) {
        for (int h = 0; h < HOURS; h++) {
            total_cost += chromosome[a][h] * appliances[a].power * tariff[h];
        }
    }
    return total_cost;
}

// Roulette selection
Chromosome ga_selection(const vector<Chromosome> &population, const vector<double>& tariff) {
    vector<double> weights;
    weights.reserve(population.size());
    for (auto &ch : population) {
        weights.push_back(1.0 / (ga_fitness(ch, tariff) + 1e-9));
    }
    double sum = accumulate(weights.begin(), weights.end(), 0.0);
    double pick = ((double) rand() / RAND_MAX) * sum;
    double cur = 0.0;

    for (int i = 0; i < (int)population.size(); i++) {
        cur += weights[i];
        if (cur >= pick) return population[i];
    }
    return population.back();
}

// One-point crossover on appliance index
pair<Chromosome, Chromosome> ga_crossover(const Chromosome &p1, const Chromosome &p2) {
    if (((double) rand() / RAND_MAX) < CROSSOVER_RATE) {
        int point = rand() % (int)appliances.size();
        Chromosome c1, c2;
        for (int i = 0; i < (int)appliances.size(); i++) {
            if (i < point) { c1.push_back(p1[i]); c2.push_back(p2[i]); }
            else           { c1.push_back(p2[i]); c2.push_back(p1[i]); }
        }
        return {c1, c2};
    }
    return {p1, p2};
}

// Mutation: resample one appliance's contiguous block
Chromosome ga_mutate(Chromosome ch) {
    if (((double) rand() / RAND_MAX) < MUTATION_RATE) {
        int idx = rand() % (int)appliances.size();
        int dur = appliances[idx].duration;
        int start = rand() % (HOURS - dur + 1);
        ch[idx] = schedule_from_start(start, dur);
    }
    return ch;
}

// Convert GA chromosome to integer starts
vector<int> ga_chromosome_to_starts(const Chromosome& ch) {
    vector<int> starts(appliances.size(), 0);
    for (int a = 0; a < (int)appliances.size(); ++a) {
        int first = -1;
        for (int h = 0; h < HOURS; ++h) {
            if (ch[a][h] == 1) { first = h; break; }
        }
        if (first < 0) first = 0;
        starts[a] = first;
    }
    return starts;
}

// Run GA, return best starts + cost
pair<vector<int>, double> run_GA(const vector<double>& tariff, bool verbose=true) {
    vector<Chromosome> population;
    population.reserve(GA_POP_SIZE);
    for (int i = 0; i < GA_POP_SIZE; i++) population.push_back(ga_createChromosome());

    Chromosome best = population[0];
    double bestCost = ga_fitness(best, tariff);

    for (int gen = 0; gen < GA_GENERATIONS; gen++) {
        vector<Chromosome> newpop;
        newpop.reserve(GA_POP_SIZE);
        for (int i = 0; i < GA_POP_SIZE / 2; i++) {
            Chromosome p1 = ga_selection(population, tariff);
            Chromosome p2 = ga_selection(population, tariff);
            auto kids = ga_crossover(p1, p2);
            Chromosome c1 = ga_mutate(kids.first);
            Chromosome c2 = ga_mutate(kids.second);
            newpop.push_back(c1);
            newpop.push_back(c2);
        }
        population.swap(newpop);

        for (auto &ch : population) {
            double f = ga_fitness(ch, tariff);
            if (f < bestCost) { bestCost = f; best = ch; }
        }
        if (verbose) cout << "GA Gen " << gen+1 << ": Best Cost = " << bestCost << "\n";
    }

    vector<int> bestStarts = ga_chromosome_to_starts(best);
    return {bestStarts, bestCost};
}

// ========================= AOA (seeded with GA best) =========================
struct Bounds { int lo, hi; }; // inclusive integer
vector<Bounds> aoa_bounds() {
    vector<Bounds> b;
    b.reserve(appliances.size());
    for (auto &ap : appliances) b.push_back({0, HOURS - ap.duration});
    return b;
}

inline double urand() { return (double)rand() / (double)RAND_MAX; }

void discretize_round_clamp(const vector<double>& x, vector<int>& start, const vector<Bounds>& B) {
    start.resize(x.size());
    for (int i = 0; i < (int)x.size(); ++i) {
        int v = (int)llround(x[i]);
        if (v < B[i].lo) v = B[i].lo;
        if (v > B[i].hi) v = B[i].hi;
        start[i] = v;
    }
}

// AOA params
const int AOA_POP = 20;
const int AOA_ITER = 60;
const double C1 = 2.0, C2 = 6.0, C3 = 2.0, C4 = 1.0, U = 0.9, L = 0.1;

// Run AOA; if seedProvided==true, X[0] is initialized from seedStart
pair<vector<int>, double> run_AOA(const vector<double>& tariff, const vector<int>& seedStart, bool seedProvided, bool verbose=true) {
    const int D = (int)appliances.size();
    const vector<Bounds> B = aoa_bounds();

    vector<vector<double>> X(AOA_POP, vector<double>(D));
    vector<vector<double>> DEN(AOA_POP, vector<double>(D));
    vector<vector<double>> VOL(AOA_POP, vector<double>(D));
    vector<vector<double>> ACC(AOA_POP, vector<double>(D));

    auto rand_in_bounds = [&](int d){
        return B[d].lo + urand() * (B[d].hi - B[d].lo);
    };

    // init
    for (int i = 0; i < AOA_POP; ++i) {
        for (int d = 0; d < D; ++d) {
            X[i][d]   = rand_in_bounds(d);
            DEN[i][d] = urand();
            VOL[i][d] = urand();
            ACC[i][d] = urand();
        }
    }
    // seed the best individual with GA result (if provided)
    if (seedProvided) {
        for (int d = 0; d < D; ++d) X[0][d] = (double)seedStart[d];
    }

    // evaluate initial best
    double bestCost = 1e100;
    vector<double> Xbest(D), DENbest(D), VOLbest(D), ACCbest(D);
    for (int i = 0; i < AOA_POP; ++i) {
        vector<int> s;
        discretize_round_clamp(X[i], s, B);
        double f = cost_from_starts(s, tariff);
        if (f < bestCost) {
            bestCost = f; Xbest = X[i]; DENbest = DEN[i]; VOLbest = VOL[i]; ACCbest = ACC[i];
        }
    }

    vector<vector<double>> DENn(AOA_POP, vector<double>(D));
    vector<vector<double>> VOLn(AOA_POP, vector<double>(D));
    vector<vector<double>> ACCn(AOA_POP, vector<double>(D));
    vector<vector<double>> ACCnorm(AOA_POP, vector<double>(D));

    // main loop
    for (int t = 1; t <= AOA_ITER; ++t) {
        // 1) dens/vol to best
        for (int i = 0; i < AOA_POP; ++i) {
            for (int d = 0; d < D; ++d) {
                DENn[i][d] = DEN[i][d] + urand() * (DENbest[d] - DEN[i][d]);
                VOLn[i][d] = VOL[i][d] + urand() * (VOLbest[d] - VOL[i][d]);
            }
        }

        // 2) transfer and decay
        double TF = exp((double(t) - AOA_ITER) / (double)AOA_ITER);
        double dens_decay = exp((double(AOA_ITER) - t) / (double)AOA_ITER) - (double)t / (double)AOA_ITER;

        // 3) acceleration
        for (int i = 0; i < AOA_POP; ++i) {
            if (TF <= 0.5) {
                int mr = rand() % AOA_POP;
                for (int d = 0; d < D; ++d) {
                    double num  = DEN[mr][d] + VOL[mr][d] * ACC[mr][d];
                    double deno = DENn[i][d] + VOLn[i][d];
                    ACCn[i][d] = num / (deno + 1e-12);
                }
            } else {
                for (int d = 0; d < D; ++d) {
                    double num  = DENbest[d] + VOLbest[d] * ACCbest[d];
                    double deno = DENn[i][d] + VOLn[i][d];
                    ACCn[i][d] = num / (deno + 1e-12);
                }
            }
        }

        // 4) normalize acc
        for (int d = 0; d < D; ++d) {
            double mn = 1e100, mx = -1e100;
            for (int i = 0; i < AOA_POP; ++i) {
                mn = min(mn, ACCn[i][d]);
                mx = max(mx, ACCn[i][d]);
            }
            double denom = (mx - mn) + 1e-12;
            for (int i = 0; i < AOA_POP; ++i) {
                double z = (ACCn[i][d] - mn) / denom;
                ACCnorm[i][d] = U * z + L;
            }
        }

        // 5) update positions
        for (int i = 0; i < AOA_POP; ++i) {
            if (TF <= 0.5) {
                int rr = rand() % AOA_POP;
                for (int d = 0; d < D; ++d) {
                    double step =  C1 * urand() * ACCnorm[i][d] * dens_decay * (X[rr][d] - X[i][d]);
                    X[i][d] += step;
                    if (X[i][d] < B[d].lo) X[i][d] = B[d].lo;
                    if (X[i][d] > B[d].hi) X[i][d] = B[d].hi;
                }
            } else {
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

        // 6) accept and update best
        DEN.swap(DENn); VOL.swap(VOLn); ACC.swap(ACCn);
        for (int i = 0; i < AOA_POP; ++i) {
            vector<int> s;
            discretize_round_clamp(X[i], s, B);
            double f = cost_from_starts(s, tariff);
            if (f < bestCost) { bestCost = f; Xbest = X[i]; DENbest = DEN[i]; VOLbest = VOL[i]; ACCbest = ACC[i]; }
        }

        if (verbose) cout << "AOA Iter " << t << ": Best Cost = " << bestCost << "\n";
    }

    vector<int> bestStart;
    discretize_round_clamp(Xbest, bestStart, B);
    return {bestStart, bestCost};
}

// ========================= Main: Hybrid GA → AOA =========================
int main() {
    srand((unsigned)time(0));

    // 1) Read tariff or use defaults
    vector<double> tariff = tariff_default;
    cout << "Use default tariff (y/n)? ";
    char yn; cin >> yn;
    if (yn == 'n' || yn == 'N') {
        cout << "Enter 24 tariff values (space-separated):\n";
        tariff.resize(HOURS);
        for (int i = 0; i < HOURS; ++i) cin >> tariff[i];
    } else {
        cout << "Using default tariff.\n";
    }
    cout << "\n";

    // 2) GA phase
    pair<vector<int>, double> ga_result = run_GA(tariff, true);
    vector<int> ga_best_starts = ga_result.first;
    double ga_best_cost = ga_result.second;

    cout << "\n=== GA Best Schedule ===\n";
    print_schedule(ga_best_starts);
    cout << "GA Best Cost = " << ga_best_cost << "\n\n";

    // 3) AOA phase (seed with GA)
    pair<vector<int>, double> aoa_result = run_AOA(tariff, ga_best_starts, true, true);
    vector<int> aoa_best_starts = aoa_result.first;
    double aoa_best_cost = aoa_result.second;

    cout << "\n=== AOA Best Schedule (seeded by GA) ===\n";
    print_schedule(aoa_best_starts);
    cout << "AOA Best Cost = " << aoa_best_cost << "\n\n";

    // 4) Final report
    vector<int> final_starts = (aoa_best_cost <= ga_best_cost) ? aoa_best_starts : ga_best_starts;
    double final_cost = min(aoa_best_cost, ga_best_cost);

    cout << "================ FINAL BEST ================\n";
    print_schedule(final_starts);
    cout << "Final Minimum Electricity Cost = " << final_cost << "\n";

    return 0;
}
