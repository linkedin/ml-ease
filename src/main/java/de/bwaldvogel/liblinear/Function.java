

package de.bwaldvogel.liblinear;

// origin: tron.h
public interface Function {

    double fun(double[] w);

    void grad(double[] w, double[] g);

    void Hv(double[] s, double[] Hs);

    int get_nr_variable();
}
