# Makefile written by L. Garrido

LIBS=-lm
INCL=

COPTS=-O3 -Wall -Werror -std=c99 -fPIC
TARGET=libcac.so

FILES_C = $(shell ls *.c)
FILES_O = $(subst .c,.o,$(FILES_C))
FILES_H = $(addprefix proto/,$(subst .c,.h,$(FILES_C)))

$(TARGET): api_cac.h $(FILES_O)
	gcc $(FILES_O) -o $(TARGET) $(LIBS) -shared 

api_prototypes.h: $(FILES_C) $(FILES_H)
	cat $(FILES_H) > api_prototypes.h
	ctags --fields=+aS --extra=+q *.c *.h

proto/%.h: %.c
	cproto $(INCL) $< -o $@ 

%.o: %.c api_cac.h Makefile
	gcc $(COPTS) $(INCL) -c $< 

all:  $(TARGET)

clean:
	/bin/rm -f *.*~ $(FILES_O) $(TARGET)

