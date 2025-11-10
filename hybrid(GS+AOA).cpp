#include <bits/stdc++.h>
using namespace std;

const int HOURS = 24;

// Representative Time-of-Use hourly tariff (Rs / kWh)
// Source: design based on Indian ToD policy examples & typical retail levels.
// - Lower during daytime solar hours and late-night off-peak
// - Higher during evening peak (18:00-22:00)
vector<double> tariff_default = {
    // 0  1   2   3   4   5
      3.0,3.0,3.0,3.0,3.0,3.0,
    // 6  7   8   9  10  11
      4.5,5.0,5.0,5.0,5.0,5.0,
    // 12 13  14  15  16  17
      4.0,4.0,4.0,4.0,5.0,6.0,
    // 18 19  20  21
     12.0,12.0,11.0,10.0,
    // 22 23
      4.0,3.5
};

// Appliance structure (duration = contiguous hours, power = kW)
// Values chosen from CLASP (ownership/use patterns) and typical power tables (BEE/CESC/industry).
// See citations below for the source documents that justify these ranges.
struct Appliance {
    string name;
    int duration;   // contiguous hours (typical single run duration or continuous operation)
    double power;   // kW (typical/average)
};

// Realistic household appliance list for an urban Indian home (representative)
vector<Appliance> appliances = {
    // name,                duration (h), power (kW)
    {"Refrigerator",         24,         0.15},   // continuous duty: 150W average. (CLASP: fridge high penetration + monitoring). :contentReference[oaicite:6]{index=6}
    {"Air Conditioner",       6,         1.2},    // typical split AC average draw (energy efficient model), runtime varies with seasons. :contentReference[oaicite:7]{index=7}
    {"Electric Vehicle (EV) Charging", 5,    3.3}, // Level-1/Level-2 charger mix: ~3.3 kW common in residential chargers. (user may vary). :contentReference[oaicite:8]{index=8}
    {"Water Heater (Geyser)", 1,         2.5},    // short run (heating), element ~2–3 kW typical. :contentReference[oaicite:9]{index=9}
    {"Washing Machine",       1,         0.5},    // typical cycle uses ~0.4–0.6 kW average (spin/heating excluded). :contentReference[oaicite:10]{index=10}
    {"Dishwasher",            1,         1.2},    // typical cycle 1–2 kW for water heating + motor. (if present) :contentReference[oaicite:11]{index=11}
    {"Clothes Dryer",        2,          2.0},    // dryer cycles 1.5–3 kW; many Indian homes don’t have dryers (optional). :contentReference[oaicite:12]{index=12}
    {"Microwave Oven",       0,          0.9},    // short bursts; set duration to 0 if you model as single minutes. (here 0 -> handle separately) :contentReference[oaicite:13]{index=13}
    {"Ironing (Iron)",       0,          1.0},    // short bursts; keep 0 for minute-level tasks (or 1 hour if you prefer). :contentReference[oaicite:14]{index=14}
    {"Room Heater / Fan (seasonal)", 4,    1.2}    // for heating in winter or fan loads; vary by season. :contentReference[oaicite:15]{index=15}
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
const int GA_POP_SIZE   = 16;    // slightly larger population
const int GA_GENERATIONS= 50;    // more generations for bigger search space
const double CROSSOVER_RATE = 0.85;
const double MUTATION_RATE  = 0.25;

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
    chromosome.reserve(appliances.size());
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
const int AOA_POP = 28;      // larger for more exploration
const int AOA_ITER = 50;     // more iterations
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
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    srand((unsigned)time(0));

    // ✅ No input required, always use default tariff
    vector<double> tariff = tariff_default;
    cout << "Using default tariff.\n\n";

    // 1) GA phase
    pair<vector<int>, double> ga_result = run_GA(tariff, true);
    vector<int> ga_best_starts = ga_result.first;
    double ga_best_cost = ga_result.second;

    cout << "\n=== GA Best Schedule ===\n";
    print_schedule(ga_best_starts);
    cout << "GA Best Cost = " << ga_best_cost << "\n\n";

    // 2) AOA phase (seed with GA)
    pair<vector<int>, double> aoa_result = run_AOA(tariff, ga_best_starts, true, true);
    vector<int> aoa_best_starts = aoa_result.first;
    double aoa_best_cost = aoa_result.second;

    cout << "\n=== AOA Best Schedule (seeded by GA) ===\n";
    print_schedule(aoa_best_starts);
    cout << "AOA Best Cost = " << aoa_best_cost << "\n\n";

    // 3) Final report
    vector<int> final_starts = (aoa_best_cost <= ga_best_cost) ? aoa_best_starts : ga_best_starts;
    double final_cost = min(aoa_best_cost, ga_best_cost);

    cout << "================ FINAL BEST ================\n";
    print_schedule(final_starts);
    cout << "Final Minimum Electricity Cost = " << final_cost << "\n";

    return 0;
}

