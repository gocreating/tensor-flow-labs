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
  		feature::list().push_back(new pattern<6>(0, 1, 4, 5, 8, 12));
  		feature::list().push_back(new pattern<6>(1, 2, 5, 6, 9, 13));
  //--------------------------------------------------
  	}

  	static int get_best_move(state s) {			// return best move dir
  //-------------TO DO--------------------------------
  		std::vector<float> returns;
  		int rewards[4];
  		for (int dir = 0; dir < 4; dir++) {
  			state current_state = s;
  			rewards[dir] = current_state.move(dir);
  			state sp = current_state;
  			float evaluatedReturn = 0.0;

  			if (rewards[dir] != -1) {
  				evaluatedReturn = rewards[dir] + sp.evaluate_score();
  			} else {
  				evaluatedReturn = -9999999999999999;
  			}
  			returns.push_back(evaluatedReturn);
  		}

  		float maxReturn = -999999999999999;
  		int bestDir = -1;

  		for (int dir = 0; dir < 4; dir++) {
  			if (returns[dir] > maxReturn) {
  				maxReturn = returns[dir];
  				bestDir = dir;
  			}
  		}

  		return bestDir;
  //--------------------------------------------------
  	}

  	static void update_tuple_values(std::vector<experience> eb, float learning_rate) {
  		for (int i = eb.size() - 1; i >= 0; i--) {
  			float error = 0.0;
  			state& sp = eb[i].sp;
  			state& spp = eb[i].spp;
  //-------------TO DO--------------------------------
  			if (i == eb.size() - 1) {					// the last experience!!

  			}
  			else {
  				int dirNext = get_best_move(spp);
  				spp.move(dirNext);
  				state spNext = spp;
  				float rewardNext = spNext.get_reward();
  				float valueSpNext = spNext.evaluate_score();
  				float valueSp = sp.evaluate_score();
  				error = rewardNext + valueSpNext - valueSp;
  			}
  //--------------------------------------------------
  			for (size_t i = 0; i < feature::list().size(); i++)
  				feature::list()[i]->update(sp.get_board(), error * learning_rate);
  		}
  	}
};
