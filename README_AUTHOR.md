# ofxC2tf2
----------

Folder structure
----------------

    of_preRelease/
      addons/
        addons_config.mk
        ofxMyAddon/
          docs/
              Doxyfile
              ...
          test/
              ...
          src/
            ofxMyAddon.h
            ofxMyAddon.cpp
            ...
          libs/
            necessaryLib/
              src/
                lib_implementation.h
                lib_implementation.cpp
                ...
              includes/
                libwhatever.h
                ...
              lib/
                osx/
                  static_libwhatever.a
                linux/
                  static_libwhatever.a
                ... //other platforms
          example_anExample/
            src/
              main.cpp
              testApp.h
              testApp.cpp
              ... //other source
            MyAddonExample.xcodeproj
            ... //other project files for other platforms
          bin/
            data/
              necessaryAsset.txt



