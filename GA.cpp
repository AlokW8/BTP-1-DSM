#include <bits/stdc++.h>
using namespace std;

const int HOURS = 24;

// ========================= CPP Tariff (Critical Peak Pricing) =========================
// Rs / kWh for each hour (0–23)
vector<double> tariff_CPP = {
    30,30,30,30,30,             // 0–4 base
    140,140,140,140,140,140,    // 5–10 critical peak
    30,32,36,44,56,60,          // 11–16 evening high
    52,46,40,36,32,30,30        // 17–23
};

// ========================= Appliance data with type & constraints =========================
// type: 0 = fixed, 1 = non-shiftable (limited window), 2 = shiftable
struct Appliance {
    string name;
    int duration;   // contiguous hours
    double power;   // kW
    int type;       // 0 = fixed, 1 = non-shiftable, 2 = shiftable
    int baseline;   // baseline start (for fixed and as center of window)
    int window;     // half-window size for non-shiftable (ignored for fixed)
    Appliance(string n, int d, double p, int tp, int b, int w = 3)
        : name(n), duration(d), power(p), type(tp), baseline(b), window(w) {}
};

// Realistic household appliances (urban Indian home)
vector<Appliance> appliances = {
    {"Air Conditioner",      4,   2.0,   2, 18, 3},
    {"Air Purifier",         6,   0.1,   2,  0, 3},
    {"Coffee Maker",         1,   0.4,   1,  7, 2},
    {"Computer",             4,   0.25,  1, 20, 3},
    {"Digital Clock",       24,   0.0025,0,  0, 0},
    {"Dishwasher",           3,   1.33,  2, 21, 4},
    {"Hair Dryer",           1,   2.0,   1,  7, 2},
    {"Electric Iron",        1,   2.0,   1,  7, 2},
    {"EV Charger",           8,   4.0,   2, 22, 6},
    {"Exhaust Fan",          2,   0.1,   2,  8, 3},
    {"Fan",                  8,   0.1,   2, 20, 4},
    {"Food Blender",         1,   0.4,   1,  8, 2},
    {"Induction Cooker",     1,   2.0,   1, 19, 2},
    {"LED Lights",           5,   0.08,  1, 19, 4},
    {"Microwave",            1,   1.5,   1, 12, 2},
    {"Night Light",          2,   0.05,  0, 22, 0},
    {"Refrigerator",         6,   0.3,   0,  0, 0},
    {"Room Heater",          3,   2.0,   2,  6, 3},
    {"Router WiFi",         24,   0.025, 0,  0, 0},
    {"Shaver",               1,   0.05,  1,  6, 2},
    {"Television",           3,   0.2,   1, 20, 3},
    {"Vacuum Cleaner",       1,   1.0,   2, 10, 8},
    {"Washing Machine",      2,   0.6,   2,  9, 8},
    {"Water Heater",         1,   2.5,   1,  6, 2},
    {"Water Pump",           1,   1.5,   1,  5, 2}
};

// ========================= Helper: bounds per appliance (by type) =========================
struct Bounds { int lo, hi; }; // inclusive

Bounds get_bounds_for_appliance(int idx) {
    const Appliance &ap = appliances[idx];
    int dur = ap.duration;
    int lo, hi;

    if (ap.type == 0) {
        // Fixed load: always at baseline, clamp to valid
        int s = min(ap.baseline, HOURS - dur);
        if (s < 0) s = 0;
        lo = hi = s;
    } else if (ap.type == 1) {
        // Non-shiftable: limited window around baseline
        lo = max(0, ap.baseline - ap.window);
        hi = min(HOURS - dur, ap.baseline + ap.window);
    } else {
        // Shiftable: can be anywhere in 0..24-duration
        lo = 0;
        hi = HOURS - dur;
    }
    if (hi < lo) hi = lo; // safety
    return {lo, hi};
}

// Cost from integer starts
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

// Build hourly load profile (kW) from start times
vector<double> load_profile_from_starts(const vector<int>& start) {
    vector<double> load(HOURS, 0.0);
    for (int a = 0; a < (int)appliances.size(); a++) {
        int s = start[a];
        int d = appliances[a].duration;
        double p = appliances[a].power;
        for (int h = s; h < s + d; ++h) {
            load[h] += p;
        }
    }
    return load;
}

// Peak-to-average ratio (PAR)
double compute_PAR(const vector<int>& start) {
    vector<double> load = load_profile_from_starts(start);
    double peak = *max_element(load.begin(), load.end());
    double sum = accumulate(load.begin(), load.end(), 0.0);
    double avg = sum / HOURS;
    return peak / (avg + 1e-12);
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

// ========================= GA (only) =========================
const int GA_POP_SIZE   = 16;
const int GA_GENERATIONS= 50;
const double CROSSOVER_RATE = 0.85;
const double MUTATION_RATE  = 0.25;

using Chromosome = vector<vector<int>>; // per-appliance 24-length 0/1 schedule

// Build schedule array (0/1) from a start time & duration
vector<int> schedule_from_start(int start, int dur) {
    vector<int> v(HOURS, 0);
    for (int i = start; i < start + dur; ++i) v[i] = 1;
    return v;
}

// Create random valid chromosome (per-appliance contiguous block respecting type/window)
Chromosome ga_createChromosome() {
    Chromosome chromosome;
    chromosome.reserve(appliances.size());
    for (int i = 0; i < (int)appliances.size(); ++i) {
        const Appliance &ap = appliances[i];
        Bounds b = get_bounds_for_appliance(i);
        int dur = ap.duration;
        int start = b.lo + (rand() % (b.hi - b.lo + 1));
        chromosome.push_back(schedule_from_start(start, dur));
    }
    return chromosome;
}

// Fitness for GA (lower is better) – cost only
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

// Mutation: resample one appliance's contiguous block respecting its bounds
Chromosome ga_mutate(Chromosome ch) {
    if (((double) rand() / RAND_MAX) < MUTATION_RATE) {
        int idx = rand() % (int)appliances.size();
        const Appliance &ap = appliances[idx];

        // For fixed loads, do not mutate
        if (ap.type == 0) return ch;

        Bounds b = get_bounds_for_appliance(idx);
        int dur = ap.duration;
        int start = b.lo + (rand() % (b.hi - b.lo + 1));
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
        if (first < 0) {
            // Fallback to baseline/bounds
            Bounds b = get_bounds_for_appliance(a);
            first = b.lo;
        }
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

// ========================= Main: GA only =========================
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    srand((unsigned)time(0));

    vector<double> tariff = tariff_CPP;
    cout << "Using CPP tariff (Critical Peak Pricing).\n\n";

    // GA phase
    auto ga_result = run_GA(tariff, true);
    vector<int> ga_best_starts = ga_result.first;
    double ga_best_cost = ga_result.second;

    cout << "\n=== GA Best Schedule ===\n";
    print_schedule(ga_best_starts);
    cout << "GA Best Cost = " << ga_best_cost << "\n\n";

    // PAR for GA
    double ga_PAR = compute_PAR(ga_best_starts);

    // Summary table (GA only)
    cout << fixed << setprecision(2);
    cout << "\n==================== RESULT SUMMARY TABLE ====================\n";
    cout << left << setw(20) << "Method"
         << setw(20) << "Cost (Rs)"
         << setw(15) << "PAR"
         << "Comment\n";
    cout << "---------------------------------------------------------------\n";

    cout << left << setw(20) << "GA"
         << setw(20) << ga_best_cost
         << setw(15) << ga_PAR
         << "Genetic Algorithm best\n";

    cout << "---------------------------------------------------------------\n\n";

    return 0;
}
