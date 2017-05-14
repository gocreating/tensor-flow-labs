#pragma once
#include <iostream>
#include <algorithm>
#include <vector>
#include <limits>
#include <cstdarg>
#include <string>
#include <sstream>
#include <cmath>
#include <fstream>
#include "2048.h"

class experience {
public:
    state sp;
    state spp;
};

int argMax(std::vector<float> vec) {
	return std::max_element(vec.begin(), vec.end()) - vec.begin();
}

class AI {
public:
    static void load_tuple_weights() {
        std::string filename = "0556508.weight";                   // put the name of weight file here
        std::ifstream in;
        in.open(filename.c_str(), std::ios::in | std::ios::binary);
        if (in.is_open()) {
            for (size_t i = 0; i < feature::list().size(); i++) {
                in >> *(feature::list()[i]);
                std::cout << feature::list()[i]->name() << " is loaded from " << filename << std::endl;
            }
            in.close();
        }
    }

    static void set_tuples() {
  //-------------TO DO--------------------------------
  		// set tuple features
  		// example: feature::list().push_back(new pattern<4>(0, 1, 2, 3));
  		feature::list().push_back(new pattern<4>(0, 4, 8, 12));
  		feature::list().push_back(new pattern<4>(1, 5, 9, 13));
  		feature::list().push_back(new pattern<4>(2, 6, 10, 14));
  		feature::list().push_back(new pattern<4>(3, 7, 11, 15));
  		feature::list().push_back(new pattern<4>(0, 1, 2, 3));
  		feature::list().push_back(new pattern<4>(4, 5, 6, 7));
  		feature::list().push_back(new pattern<4>(8, 9, 10, 11));
  		feature::list().push_back(new pattern<4>(12, 13, 14, 15));
  //--------------------------------------------------
  	}

    static float approximateValue(state s) {
  		float approximatedValue = 0;
  		for (size_t i = 0; i < feature::list().size(); i++) {
  			approximatedValue += feature::list()[i] -> estimate(s.get_board());
  		}
  		return approximatedValue;
  	}

  	static int get_best_move(state s) {			// return best move dir
  //-------------TO DO--------------------------------
  		std::vector<float> evaluatedReturns;
  		for (int dir = 0; dir < 4; dir++) {
  			float reward = s.move(dir);
  			float approximatedValue = approximateValue(s);
  			evaluatedReturns.push_back(reward + approximatedValue);
  		}
  		int maxReturnDir = argMax(evaluatedReturns);
  		return maxReturnDir;
  //--------------------------------------------------
  	}

};
