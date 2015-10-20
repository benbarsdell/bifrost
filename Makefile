
all: bifrost std_plugins

bifrost:
	$(MAKE) -C ./src all
.PHONY: bifrost

std_plugins:
	$(MAKE) -C ./plugins/std all
.PHONY: std_plugins

clean:
	$(MAKE) -C ./src clean
	$(MAKE) -C ./plugins/std clean
	rm -f doc/html/*
	rm -f doc/latex/*
.PHONY: clean

doc/html/index.html:
	doxygen
doc: doc/html/index.html
.PHONY: doc
