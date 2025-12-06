#include <bits/stdc++.h>
using namespace std;

const int HOURS = 24;

// ========================= Appliance data (Table 2) =========================
struct Appliance {
    string name;
    int duration;   // daily usage in hours (treated as contiguous here)
    double power;   // kW
};

// Appliances used in simulations (from Fig. / Table 2)
vector<Appliance> appliances = {
    // Shiftable load appliances
    {"Vacuum cleaner",   6, 0.7},
    {"Water heater",    12, 5.0},
    {"Toaster",          3, 1.5},
    {"Water pump",       8, 1.0},
    {"Geyser",           2, 3.5},
    {"Dish washer",      8, 1.8},

    // Non-shiftable load appliances
    {"Washing machine",  5, 0.7},
    {"Cloth dryer",      4, 5.0},
    {"Electric frying pot", 2, 1.2},

    // Fixed load appliances
    {"Rice cookers",     4, 1.0},
    {"Refrigerator",    18, 0.225},
    {"AC",              15, 1.5},
    {"Lights",           9, 0.25},
    {"Television",       8, 0.2},
    {"Oven",            10, 2.15}
};

// ========================= CPP Tariff only ====================
// All values are in cent/kWh, 24 elements, indices 0–23.
vector<double> tariff_CPP = {
    30,30,30,30,30,             // 0–4 base
    140,140,140,140,140,140,    // 5–10 critical peak
    30,32,36,44,56,60,          // 11–16 evening high
    52,46,40,36,32,30,30        // 17–23
};

// ========================= Cost helpers =====================================
double cost_from_starts(const vector<int>& start, const vector<double>& tariff) {
    double cost = 0.0;
    for (int a = 0; a < (int)appliances.size(); a++) {
        int s = start[a];
        int d = appliances[a].duration;
        double p = appliances[a].power;
        for (int h = s; h < s + d; ++h) {
            cost += p * tariff[h];
        }
    }
    return cost;
}

void print_schedule(const vector<int>& start) {
    for (int a = 0; a < (int)appliances.size(); a++) {
        cout << appliances[a].name << ": ";
        for (int h = start[a]; h < start[a] + appliances[a].duration; ++h) {
            cout << h << " ";
        }
        cout << "\n";
    }
}

/************ NEW: hourly load profile & PAR ************************/

// Build hourly load profile P_Lsch(t) [kW] from start times
vector<double> hourly_load_from_starts(const vector<int>& start) {
    vector<double> load(HOURS, 0.0);
    for (int a = 0; a < (int)appliances.size(); ++a) {
        int s = start[a];
        int d = appliances[a].duration;
        double p = appliances[a].power;
        for (int h = s; h < s + d; ++h) {
            load[h] += p;
        }
    }
    return load;
}

// Compute PAR = (max(PL)^2)/(avg(PL)^2)
double compute_PAR_from_starts(const vector<int>& start) {
    vector<double> load = hourly_load_from_starts(start);
    double peak = *max_element(load.begin(), load.end());
    double sum = accumulate(load.begin(), load.end(), 0.0);
    double avg  = sum / HOURS;
    if (avg <= 0.0) return 0.0;
    return (peak * peak) / (avg * avg);
}

/******************************************************************/

// ========================= GA (same style, small tweaks) =====================
const int GA_POP_SIZE   = 16;
const int GA_GENERATIONS= 50;
const double CROSSOVER_RATE = 0.85;
const double MUTATION_RATE  = 0.25;

using Chromosome = vector<vector<int>>; // per-appliance 24-length 0/1 schedule

vector<int> schedule_from_start(int start, int dur) {
    vector<int> v(HOURS, 0);
    for (int i = start; i < start + dur; ++i) v[i] = 1;
    return v;
}

Chromosome ga_createChromosome() {
    Chromosome chromosome;
    chromosome.reserve(appliances.size());
    for (auto &app : appliances) {
        int start = rand() % (HOURS - app.duration + 1);
        chromosome.push_back(schedule_from_start(start, app.duration));
    }
    return chromosome;
}

double ga_fitness(const Chromosome &chromosome, const vector<double>& tariff) {
    double total_cost = 0.0;
    for (int a = 0; a < (int)appliances.size(); a++) {
        for (int h = 0; h < HOURS; h++) {
            total_cost += chromosome[a][h] * appliances[a].power * tariff[h];
        }
    }
    return total_cost;
}

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

pair<Chromosome, Chromosome> ga_crossover(const Chromosome &p1, const Chromosome &p2) {
    if (((double) rand() / RAND_MAX) < CROSSOVER_RATE) {
        int point = rand() % (int)appliances.size();
        Chromosome c1, c2;
        c1.reserve(appliances.size());
        c2.reserve(appliances.size());
        for (int i = 0; i < (int)appliances.size(); i++) {
            if (i < point) { c1.push_back(p1[i]); c2.push_back(p2[i]); }
            else           { c1.push_back(p2[i]); c2.push_back(p1[i]); }
        }
        return {c1, c2};
    }
    return {p1, p2};
}

Chromosome ga_mutate(Chromosome ch) {
    if (((double) rand() / RAND_MAX) < MUTATION_RATE) {
        int idx = rand() % (int)appliances.size();
        int dur = appliances[idx].duration;
        int start = rand() % (HOURS - dur + 1);
        ch[idx] = schedule_from_start(start, dur);
    }
    return ch;
}

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

// ========================= AOA (same as before) =============================
struct Bounds { int lo, hi; };

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
const int AOA_POP = 28;
const int AOA_ITER = 50;
const double C1 = 2.0, C2 = 6.0, C3 = 2.0, C4 = 1.0, U = 0.9, L = 0.1;

pair<vector<int>, double> run_AOA(const vector<double>& tariff,
                                  const vector<int>& seedStart,
                                  bool seedProvided,
                                  bool verbose=true) {
    const int D = (int)appliances.size();
    const vector<Bounds> B = aoa_bounds();

    vector<vector<double>> X(AOA_POP, vector<double>(D));
    vector<vector<double>> DEN(AOA_POP, vector<double>(D));
    vector<vector<double>> VOL(AOA_POP, vector<double>(D));
    vector<vector<double>> ACC(AOA_POP, vector<double>(D));

    auto rand_in_bounds = [&](int d){
        return B[d].lo + urand() * (B[d].hi - B[d].lo);
    };

    for (int i = 0; i < AOA_POP; ++i) {
        for (int d = 0; d < D; ++d) {
            X[i][d]   = rand_in_bounds(d);
            DEN[i][d] = urand();
            VOL[i][d] = urand();
            ACC[i][d] = urand();
        }
    }
    if (seedProvided) {
        for (int d = 0; d < D; ++d) X[0][d] = (double)seedStart[d];
    }

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

    for (int t = 1; t <= AOA_ITER; ++t) {
        for (int i = 0; i < AOA_POP; ++i) {
            for (int d = 0; d < D; ++d) {
                DENn[i][d] = DEN[i][d] + urand() * (DENbest[d] - DEN[i][d]);
                VOLn[i][d] = VOL[i][d] + urand() * (VOLbest[d] - VOL[i][d]);
            }
        }

        double TF = exp((double(t) - AOA_ITER) / (double)AOA_ITER);
        double dens_decay = exp((double(AOA_ITER) - t) / (double)AOA_ITER) - (double)t / (double)AOA_ITER;

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
                    double step = F * C2 * urand() * ACCnorm[i][d] * dens_decay
                                  * (Tpar * Xbest[d] - X[i][d]);
                    X[i][d] = Xbest[d] + step;
                    if (X[i][d] < B[d].lo) X[i][d] = B[d].lo;
                    if (X[i][d] > B[d].hi) X[i][d] = B[d].hi;
                }
            }
        }

        DEN.swap(DENn); VOL.swap(VOLn); ACC.swap(ACCn);
        for (int i = 0; i < AOA_POP; ++i) {
            vector<int> s;
            discretize_round_clamp(X[i], s, B);
            double f = cost_from_starts(s, tariff);
            if (f < bestCost) {
                bestCost = f; Xbest = X[i]; DENbest = DEN[i]; VOLbest = VOL[i]; ACCbest = ACC[i];
            }
        }

        if (verbose) cout << "AOA Iter " << t << ": Best Cost = " << bestCost << "\n";
    }

    vector<int> bestStart;
    discretize_round_clamp(Xbest, bestStart, aoa_bounds());
    return {bestStart, bestCost};
}

// ========================= Baseline (unscheduled) ===========================
vector<int> baseline_unscheduled() {
    vector<int> s(appliances.size(), 0); // everyone starts at hour 0
    return s;
}

// ========================= Main (CPP only) ==================================
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    srand((unsigned)time(0));

    cout << fixed << setprecision(2);

    cout << "\n================ Tariff: CPP ================\n";

    // 0) Unscheduled
    vector<int> base = baseline_unscheduled();
    double cost_unscheduled = cost_from_starts(base, tariff_CPP);
    double par_unscheduled  = compute_PAR_from_starts(base);
    cout << "Unscheduled cost = " << cost_unscheduled << " cent\n";
    cout << "Unscheduled PAR  = " << par_unscheduled  << "\n\n";

    // 1) GA only
    auto ga_res = run_GA(tariff_CPP, true);
    vector<int> ga_best = ga_res.first;
    double ga_cost = ga_res.second;
    double ga_par  = compute_PAR_from_starts(ga_best);
    cout << "\nGA best cost (CPP) = " << ga_cost << " cent\n";
    cout << "GA PAR (CPP)       = " << ga_par  << "\n";

    // 2) AOA only (no GA seed)
    vector<int> dummy;
    auto aoa_res = run_AOA(tariff_CPP, dummy, false, true);
    vector<int> aoa_best = aoa_res.first;
    double aoa_cost = aoa_res.second;
    double aoa_par  = compute_PAR_from_starts(aoa_best);
    cout << "\nAOA best cost (CPP) = " << aoa_cost << " cent\n";
    cout << "AOA PAR (CPP)       = " << aoa_par  << "\n";

    // 3) Hybrid GA → AOA (HAG)
    auto hag_res = run_AOA(tariff_CPP, ga_best, true, true);
    vector<int> hag_best = hag_res.first;
    double hag_cost = hag_res.second;
    double hag_par  = compute_PAR_from_starts(hag_best);
    cout << "\nHybrid GA→AOA (HAG) best cost (CPP) = "
         << hag_cost << " cent\n";
    cout << "HAG PAR (CPP)                      = "
         << hag_par  << "\n";

    // Final small tables
    cout << "\n\n========== Result Table (Total electricity cost, cent) ==========\n";
    cout << left << setw(8) << "Tariff"
         << setw(15) << "Unscheduled"
         << setw(10) << "GA"
         << setw(10) << "AOA"
         << setw(10) << "HAG" << "\n";

    cout << left << setw(8) << "CPP"
         << setw(15) << cost_unscheduled
         << setw(10) << ga_cost
         << setw(10) << aoa_cost
         << setw(10) << hag_cost << "\n";

    cout << "\n========== PAR Table (dimensionless) ==========\n";
    cout << left << setw(8) << "Tariff"
         << setw(15) << "Unscheduled"
         << setw(10) << "GA"
         << setw(10) << "AOA"
         << setw(10) << "HAG" << "\n";

    cout << left << setw(8) << "CPP"
         << setw(15) << par_unscheduled
         << setw(10) << ga_par
         << setw(10) << aoa_par
         << setw(10) << hag_par << "\n";

    return 0;
}
