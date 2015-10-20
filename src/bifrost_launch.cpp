
#include "Pipeline.hpp"

#include <iostream>
using std::cout;
using std::endl;
#include <csignal>

template<class C>
class Singleton {
public:
	static C& instance() {
		static C inst;
		return inst;
	}
protected:
	Singleton() {}
private:
	Singleton(Singleton const& );
	Singleton& operator=(Singleton const& );
};

class Application : public Singleton<Application> {
	friend class Singleton<Application>;
	static void signal_handler(int sig, siginfo_t* si, void* context) {
		printf("Received signal: %s\n", strsignal(sig));
		Application* self = &Application::instance();
		self->_pipeline.shutdown();
	}
	Application() {
		// Install signal handlers
		struct sigaction sa;
		sa.sa_flags = SA_SIGINFO;
		sigemptyset(&sa.sa_mask);
		sa.sa_sigaction = Application::signal_handler;
		// ** TODO: Include all of these when done testing
		//int signals[] = {SIGHUP, SIGINT, SIGQUIT, SIGTERM, SIGTSTP, 0};
		int signals[] = {SIGINT, 0};
		//int signals[] = {0};
		for( int* sig=signals; *sig!=0; ++sig ) {
			if( sigaction(*sig, &sa, NULL ) == -1) {
				std::cout << "WARNING: Failed to set handler for signal " << *sig << std::endl;
			}
		}
	}
	void parse_command_line(List&   positional_args,
	                        Object& keyword_args,
	                        int argc, char* argv[],
	                        std::string keyvalsep="=") {
		for( int i=1; i<argc; ++i ) {
			std::string arg = argv[i];
			size_t split = arg.find(keyvalsep);
			if( split == std::string::npos ) {
				positional_args.push_back(parse_value(arg));
			}
			else {
				std::string key  = arg.substr(0, split);
				std::string sval = arg.substr(split+keyvalsep.size());
				keyword_args.insert(std::make_pair(key, parse_value(sval)));
			}
		}
	}
	void print_usage(int argc, char* argv[]) {
		std::cout << "Usage: "
		          << argv[0]
		          << " pipeline_definition.json "
		          << "[kwarg1=val1 ...]"
		          << std::endl;
	}
public:
	int main(int argc, char* argv[]) {
		List   positional_args;
		Object keyword_args;
		parse_command_line(positional_args, keyword_args, argc, argv);
		if( positional_args.size() < 1 ||
		    !positional_args[0].is<std::string>() ) {
			print_usage(argc, argv);
			return -1;
		}
		std::string pipeline_def_file = positional_args[0].get<std::string>();
		
		_pipeline.load(pipeline_def_file, keyword_args);
		_pipeline.launch();
		_pipeline.wait();
		
		cout << "All done." << endl;
		return 0;
	}
private:
	Pipeline _pipeline;
};

int main(int argc, char* argv[]) {
	return Application::instance().main(argc, argv);
}
