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

// ========================= AOA (ONLY) =========================
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

// Run AOA; seedProvided==false → pure AOA (no GA seeding)
pair<vector<int>, double> run_AOA(const vector<double>& tariff,
                                  const vector<int>& seedStart,
                                  bool seedProvided,
                                  bool verbose=true) {
    const int D = (int)appliances.size();

    vector<Bounds> B(D);
    for (int d = 0; d < D; ++d) B[d] = get_bounds_for_appliance(d);

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
    // optional seeding (not used in AOA-only call)
    if (seedProvided && (int)seedStart.size() == D) {
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
            if (f < bestCost) {
                bestCost = f; Xbest = X[i]; DENbest = DEN[i]; VOLbest = VOL[i]; ACCbest = ACC[i];
            }
        }

        if (verbose) cout << "AOA Iter " << t << ": Best Cost = " << bestCost << "\n";
    }

    vector<int> bestStart;
    discretize_round_clamp(Xbest, bestStart, B);
    return {bestStart, bestCost};
}

// ========================= Main: AOA only =========================
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    srand((unsigned)time(0));

    vector<double> tariff = tariff_CPP;
    cout << "Using CPP tariff (Critical Peak Pricing).\n\n";

    // AOA-only phase (no GA seeding: seedProvided = false)
    vector<int> dummy_seed(appliances.size(), 0);
    auto aoa_result = run_AOA(tariff, dummy_seed, false, true);
    vector<int> aoa_best_starts = aoa_result.first;
    double aoa_best_cost = aoa_result.second;

    cout << "\n=== AOA Best Schedule ===\n";
    print_schedule(aoa_best_starts);
    cout << "AOA Best Cost = " << aoa_best_cost << "\n\n";

    // PAR for AOA
    double aoa_PAR = compute_PAR(aoa_best_starts);

    // Summary table (AOA only)
    cout << fixed << setprecision(2);
    cout << "\n==================== RESULT SUMMARY TABLE ====================\n";
    cout << left << setw(20) << "Method"
         << setw(20) << "Cost (Rs)"
         << setw(15) << "PAR"
         << "Comment\n";
    cout << "---------------------------------------------------------------\n";

    cout << left << setw(20) << "AOA"
         << setw(20) << aoa_best_cost
         << setw(15) << aoa_PAR
         << "Arithmetic Optimization best\n";

    cout << "---------------------------------------------------------------\n\n";

    return 0;
}
