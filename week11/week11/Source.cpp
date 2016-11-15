#include <iostream>
#include <vector>

// linear regression < deep learning < machine learning
class LinearHypothesis
{
public:
	// linear hypothesis : y = a * x + b
	float a_ = 0.1f;
	float b_ = 0.1f;

	float getY(const float& x_input)
	{
		return a_ * x_input + b_; // returns y = a*x+b
	}
};

class NN
{
public:
	float a0_ = 0.1f;
	float a1_ = 0.1f;
	float b0_ = 0.1f;
	float b1_ = 0.1f;

	float getY(const float& x_input)
	{
		return a1_ * (a0_ * x_input + b0_) + b1_;
	}
};

class Hypothesis
{
public:
	float a_ = 0.0f;
	float b_ = 0.0f;
	float c_ = 0.0f;

	float getY(const float& x_input)
	{
		return a_ * x_input * x_input + b_ * x_input + c_;
	}
};

const int num_data = 5;
const float input[num_data] = { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f };
const float output[num_data] = { 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };

int main()
{

	NN ly;

	for (int tr = 0; tr < 10000; tr++)
	{
		for (int i = 0; i < num_data; i++)
		{
			// let's train our linear hypothesis to answer correctly!
			const float x_input = input[i];
			const float y_output = ly.getY(x_input);
			const float y_target = output[i];
			const float error = y_output - y_target;
			
			// we can consider that our LH is trained well when error is 0 or small enough
			// we define squared error
			const float sqr_error = 0.5 * error * error; // always zero or positive

														 // sqr_error = 0.5 * (a * x + b - y_target)^2
														 // d sqr_error / da = 2*0.5*(a * x + b - y_target) * x; 
														 // d sqr_error / db = 2*0.5*(a * x + b - y_target) * 1;
			const float dse_over_da0 = error * ly.a1_ * x_input;
			const float dse_over_da1 = error * (ly.a0_* x_input + ly.b0_);
			const float dse_over_db0 = error * ly.a1_;
			const float dse_over_db1 = error;

			const float lr = 0.001;
			ly.a0_ -= dse_over_da0 * lr;
			ly.b0_ -= dse_over_db0 * lr;
			ly.a1_ -= dse_over_da1 * lr;
			ly.b1_ -= dse_over_db1 * lr;

			//std::cout << "a0: " << ly.a0_ << "\tb0: " << ly.b0_ << "\ta1: " << ly.a1_ << "\tb1: " << ly.b1_ << std::endl;

			//std::cout << "target value: " << output[0] << std::endl;
			//std::cout << "trained value: " << ly.getY(input[0]) << std::endl;
		}
	}

	std::cout << "numerical method: a1_ * (a0_ * x_input + b0_) + b1_ " << std::endl;
	std::cout << "a0: "<<ly.a0_ << "\tb0: " << ly.b0_ << "\ta1: " << ly.a1_ << "\tb1: " << ly.b1_ << std::endl;
	for (int i = 0; i < num_data; i++)
	{
		std::cout << "target: " << output[i] << std::endl;
		std::cout << "trained: " << ly.getY(input[i]) << std::endl;
	}
	

	LinearHypothesis hidden_layer;
	LinearHypothesis output_layer;

	for (int tr = 0; tr < 10000; tr++)
	{
		for (int i = 0; i < num_data; i++)
		{
			// let's train our linear hypothesis to answer correctly!
			const float x0_input = input[i];
			const float y0_output = hidden_layer.getY(x0_input);
			const float x1_input = y0_output;
			const float y1_output = output_layer.getY(x1_input);
			const float y_target = output[i];
			const float error =  y1_output - y_target;


			

			// we can consider that our LH is trained well when error is 0 or small enough
			// we define squared error
			const float sqr_error = 0.5 * error * error; // always zero or positive

														 // sqr_error = 0.5 * (a * x + b - y_target)^2
														 // d sqr_error / da = 2*0.5*(a * x + b - y_target) * x; 
														 // d sqr_error / db = 2*0.5*(a * x + b - y_target) * 1;
			const float dse_over_da0 = error * output_layer.a_ * x0_input;
			const float dse_over_da1 = error * (hidden_layer.a_*x0_input + hidden_layer.b_);
			const float dse_over_db0 = error * output_layer.a_;
			const float dse_over_db1 = error;

			const float lr = 0.01;
			hidden_layer.a_ -= dse_over_da0 * lr;
			hidden_layer.b_ -= dse_over_db0 * lr;
			output_layer.a_ -= dse_over_da1 * lr;
			output_layer.b_ -= dse_over_db1 * lr;

			//std::cout << "x0_input: "<< x0_input << "\ty0_output: " << y0_output << "\tx1_input: " << x1_input << "\ty1_output" << y1_output << "\ty_target " << y_target << "\terror: " << error << std::endl;
			//std::cout << "a0: " << hiden_layer.a_ << "\tb0: " << hiden_layer.b_ << "\ta1: " << output_layer.a_ << "\ta2: " << output_layer.b_ << std::endl;

			//std::cout << "target value: " << output[0] << std::endl;
			//std::cout << "trained value: " << output_layer.getY(hiden_layer.getY(input[0])) << std::endl;
		}
	}
	std::cout << "-------------------------------------------------------------------------------------\n";
	std::cout << "neural Network method : input -> hidden_layer -> output_layer -> output" << std::endl;
	std::cout << "a0: " << hidden_layer.a_ << "\tb0: " << hidden_layer.b_ << "\ta1: " << output_layer.a_ << "\tb1: " << output_layer.b_ << std::endl;
	for (int i = 0; i < num_data; i++)
	{
		std::cout << "target: " << output[i] << std::endl;
		std::cout << "trained: " << output_layer.getY(hidden_layer.getY(input[i])) << std::endl;
	}
	
	return 0;
}