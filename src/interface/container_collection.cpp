


  void ContainerCollection::AddContainer(const std::string& name, Container<T>& src) {
    // error check for duplicate names

    auto c = Container<T>();
    for (auto v : src._varArray) {
      if (v.isSet(Metadata::oneCopy)) {
        c._varArray.push_back(v);
      } else {
        c._varArray.push_back( std::make_shared<Variable<T>>(*v) );
      }
    }

    for (auto v : src._faceArray) {
      if (v.isSet(Metadata::oneCopy)) {
        c._faceArray.push_back(v);
      } else {
        throw std::runtime_error("Non-oneCopy face variables are not yet supported");
      }
    }

    for (auto v : src._sparseVars) {
      if (v.isSet(Metadata::oneCopy)) {
        src._sparseVars
      }
    }


  }